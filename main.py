"""
MCM 问题 C 主程序
2026 年 1 月
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入各模块
from preprocess.load_and_clean import (
    load_data,
    clean_data,
    reshape_to_long_weeks,
)
from preprocess.detect_errors import (
    detect_outliers,
    detect_score_overflow,
    detect_exit_week_mismatch,
    detect_placement_inconsistency,
    validate_ranges,
)
from preprocess.handle_special_cases import (
    handle_missing_values,
    normalize_categorical,
    mark_withdrawal_and_elimination,
)
from features.judge_features import build_judge_features
from features.environment_features import build_environment_features
from features.personal_features import build_personal_features

from models.vote_share_model import EliminationProbModel
from models.vote_rank_model import VoteRankModel
from models.season28_plus import Season28PlusRules
from evaluation.consistency import (
    elimination_boundary_margin,
    calculate_hit_at_k,
    analyze_boundary_margin_distribution
)
from evaluation.uncertainty import (
    weekly_uncertainty,
    analyze_vote_share_intervals,
    analyze_vote_count_uncertainty
)
from evaluation.fairness import reversal_ratios


class _TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    """主程序"""
    print("=" * 60)
    print("MCM 问题 C 数据分析模型")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n[步骤 1] 加载数据...")
    data_path = project_root / 'data' / '2026_MCM_Problem_C_Data.csv'
    
    if not data_path.exists():
        print(f"警告: 数据文件不存在: {data_path}")
        print("请确保数据文件已放置在正确位置")
        return
    
    raw_data = load_data(str(data_path))
    
    if raw_data is None:
        print("数据加载失败，退出程序")
        return
    
    print(f"数据形状: {raw_data.shape}")
    print(f"列名: {list(raw_data.columns)}")
    
    # 2. 数据清理
    print("\n[步骤 2] 数据清理...")
    cleaned_data = clean_data(raw_data)
   
    cleaned_data = mark_withdrawal_and_elimination(cleaned_data)


    '''
    # 导出清理后的数据
    output_path = project_root / 'data' / 'cleaned_data.csv'
    try:
        cleaned_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[完成] 清理后的数据已导出至: {output_path}")
    except PermissionError:
        print(f"[警告] 无法写入: {output_path}（文件被占用）")
    '''
    
    # 3. 错误检测
    print("\n[步骤 3] 检测异常值...")
    
    # 3.1 定义业务规则范围
    business_ranges = {
        "celebrity_age_during_season": (0, 100),  # 合理年龄范围：0-100岁
        "placement": (1, 20),                      # 排名范围：1-20名
        "season": (1, 34),                         # 赛季范围：1-34季
    }
    
    # 3.2 验证数据范围
    range_errors = validate_ranges(cleaned_data, business_ranges)
    total_range_errors = sum(len(df) for df in range_errors.values())
    print(f"数据范围异常记录总数: {total_range_errors}")
    
    for col, error_df in range_errors.items():
        if len(error_df) > 0:
            print(f"\n{col} 超出范围的记录数: {len(error_df)}")
            print(error_df[["celebrity_name", "season", col]].head(5))
    
    # 3.3 其他错误检测
    outliers = detect_outliers(cleaned_data)
    print(f"\n统计学异常值数量: {sum(len(v) for v in outliers.values())}")
    overflow = detect_score_overflow(cleaned_data)
    mismatch = detect_exit_week_mismatch(cleaned_data)
    placement_dup = detect_placement_inconsistency(cleaned_data)
    print(f"评分超过 10 的记录: {len(overflow)}")
    if len(overflow) > 0:
        print("\n出错记录（评分超过10）:")
        print(overflow[["celebrity_name", "season", "results"]].head(10))
    print(f"退出周不一致记录: {len(mismatch)}")
    if len(mismatch) > 0:
        print("\n出错记录（退出周不一致）:")
        print(mismatch[["celebrity_name", "season", "results", "exit_week_from_results", "exit_week_from_scores"]].head(10))
    print(f"名次重复记录: {len(placement_dup)}")
    if len(placement_dup) > 0:
        print("\n出错记录（名次重复）:")
        print(placement_dup.head(10))
    print(f"评分超过 10 的记录: {len(overflow)}")
    print(f"退出周不一致记录: {len(mismatch)}")
    print(f"名次重复记录(按季): {len(placement_dup)}")
    
    # 4. 构建长表
    print("\n[步骤 4] 构建周粒度数据...")
    long_df = reshape_to_long_weeks(cleaned_data)
    
     # 创建淘汰标签：该周是否被淘汰
    print("- 创建淘汰标签...")
    
    # 获取每季的最后一周
    max_week_by_season = long_df.groupby("season")["week"].max().reset_index(name="max_week")
    long_df = long_df.merge(max_week_by_season, on="season", how="left")
    
    # 冠军 (placement=1) 的 exit_week 改为 0（表示未被淘汰）
    is_champion = (long_df["placement"] == 1)
    long_df.loc[is_champion, "exit_week"] = 0
    
    # 基于调整后的exit_week创建淘汰标签
    long_df["is_eliminated"] = (long_df["week"] == long_df["exit_week"]).astype(int)
    
    # 过滤有效周（至少有一个人有评分，即 judge_total_score > 0）
    long_df = long_df[long_df["judge_total_score"] > 0].copy()
    
    # 统计每周实际淘汰人数（0、1、2人等）
    elimination_counts = long_df.groupby(["season", "week"])["is_eliminated"].sum().reset_index(name="n_eliminated")
    long_df = long_df.merge(elimination_counts, on=["season", "week"], how="left")
    long_df["n_eliminated"] = long_df["n_eliminated"].fillna(0).astype(int)
    
    print(f"有效周数: {len(elimination_counts)}")
    
    # 统计所有出现的淘汰人数
    elimination_distribution = elimination_counts['n_eliminated'].value_counts().sort_index()
    print("淘汰情况统计:")
    for n_elim, count in elimination_distribution.items():
        if n_elim == 0:
            print(f"  无淘汰: {count}周")
        elif n_elim == 1:
            print(f"  单淘汰: {count}周")
        elif n_elim == 2:
            print(f"  双淘汰: {count}周")
        elif n_elim == 3:
            print(f"  三淘汰: {count}周")
        else:
            print(f"  {n_elim}人淘汰: {count}周")
    
    # 详细列出所有多淘汰周（淘汰人数 >= 2）
    multi_elimination = elimination_counts[elimination_counts['n_eliminated'] >= 2]
    if len(multi_elimination) > 0:
        print(f"\n[多淘汰周详情] 共 {len(multi_elimination)} 周：")
        for _, row in multi_elimination.iterrows():
            season, week, n = int(row['season']), int(row['week']), int(row['n_eliminated'])
            # 获取该周被淘汰的选手
            eliminated_in_week = long_df[(long_df['season'] == season) & (long_df['week'] == week) & (long_df['is_eliminated'] == 1)]
            names = eliminated_in_week['celebrity_name'].tolist()
            print(f"  Season {season}, Week {week}: {n}人淘汰 → {', '.join(names)}")



    # 5. 特征工程
    print("\n[步骤 5] 特征工程...")
   
    
    long_df = build_judge_features(long_df)
    long_df = build_environment_features(long_df)
    long_df = build_personal_features(long_df)
    
    # 6. 模型训练
    print("\n[步骤 6] 模型训练...")
    feature_cols = [
        "judge_mean",
        "judge_std",
        "judge_mean_roll_std",
        "judge_std_roll_std",

        "env_mean",
        "env_std",
        "env_count",
        "env_special",
        "env_week",
        "env_intensity",
        "env_boundary_margin",
        "env_ties",
        "env_mean_roll_std",
        "env_std_roll_std",
        "env_count_delta",
        "env_intensity_delta",
        "env_ties_delta",
        "env_boundary_margin_delta",
        
        "weeks_in_competition",
        "rank_max_change",
        "rank_mean",
        "rank_std",
        "rank_spike_count",
        "bottom_k_count",
    ]

    long_df[feature_cols] = long_df[feature_cols].fillna(0)
   

   
    
    # 训练淘汰概率模型（预测该周是否被淘汰）
    print("- 淘汰概率模型 (预测该周是否被淘汰, 训练集: 第1-34季)")
    share_train_mask = (long_df["season"] >= 1) & (long_df["season"] <= 34) 
    elimination_model = EliminationProbModel()
    elimination_model.fit(long_df.loc[share_train_mask, feature_cols], long_df.loc[share_train_mask, "is_eliminated"])
    long_df["elimination_prob"] = elimination_model.predict(long_df[feature_cols])
    long_df["elimination_prob"] = long_df["elimination_prob"].clip(0, 1)  # 限制在 0-1
    
    # 先计算评委得分百分比
    long_df["judge_share"] = long_df.groupby(["season", "week"])["judge_total_score"].transform(
        lambda s: s / (s.sum() + 1e-10)
    )
    
    # 根据淘汰概率反推观众投票比例
    # 逻辑（百分比结合法）：
    # 1. combined_share = judge_share + audience_share
    # 2. judge_share 周内和为1，audience_share 周内和为1，所以 combined_share 周内和为2
    # 3. 淘汰概率高 → combined_share 低
    # 4. 用 survival_prob 归一化后乘以2来估计 combined_share
    print("- 基于淘汰概率和评委得分反推观众投票比例...")
    long_df["survival_prob"] = 1 - long_df["elimination_prob"]
    
    # 将存活概率归一化并缩放到和为2（对应 combined_share 的周内和）
    long_df["estimated_combined_share"] = long_df.groupby(["season", "week"])["survival_prob"].transform(
        lambda s: 2.0 * s / (s.sum() + 1e-10)
    )
    
    # 反推观众投票比例：audience_share = estimated_combined_share - judge_share
    long_df["audience_share_raw"] = long_df["estimated_combined_share"] - long_df["judge_share"]
    
    def normalize_audience_share(raw_series):
        raw = raw_series.values
        # 平移使最小值为0
        shifted = raw - raw.min()
        total = shifted.sum()
        if total > 1e-10:
            return shifted / total
        else:
            # 如果全0，则均匀分配
            return np.ones(len(raw)) / len(raw)
    
    long_df["audience_share"] = long_df.groupby(["season", "week"])["audience_share_raw"].transform(normalize_audience_share)
    
    # 排名法：先计算评委排名（1为最高分）
    long_df["judge_rank"] = long_df.groupby(["season", "week"])["judge_total_score"].rank(
        ascending=False, method="min"
    )

    # 排名法解耦：由淘汰概率反推综合排名得分
    # 1. 将 elimination_prob 归一化到和为1
    # 2. 乘以 n(n+1) 得到综合排名得分（淘汰概率越高，综合排名得分越高，即排名越差）
    def calc_combined_rank_score(group):
        elim_prob = group["elimination_prob"].values
        n = len(group)
        # 归一化并缩放到 n(n+1)
        normalized = elim_prob / (elim_prob.sum() + 1e-10)
        return normalized * n * (n + 1)
    
    long_df["combined_rank_score"] = long_df.groupby(["season", "week"]).apply(
        lambda g: pd.Series(calc_combined_rank_score(g), index=g.index)
    ).reset_index(level=[0, 1], drop=True)

    # 观众排名得分 = 综合排名得分 - 评委排名
    long_df["audience_rank_score"] = long_df["combined_rank_score"] - long_df["judge_rank"]

    # 将观众排名得分映射到 1..n 的排名（1 为最高）
    long_df["audience_rank"] = long_df.groupby(["season", "week"])["audience_rank_score"].rank(
        ascending=True, method="min"
    )

    # 百分比结合法：综合得分 = 评委百分比 + 观众百分比
    long_df["combined_share"] = long_df["judge_share"] + long_df["audience_share"]

    # 百分比结合法：排名（1为最好）
    long_df["audience_share_rank"] = long_df.groupby(["season", "week"])["audience_share"].rank(
        ascending=False, method="min"
    )
    long_df["combined_share_rank"] = long_df.groupby(["season", "week"])["combined_share"].rank(
        ascending=False, method="min"
    )

    # 排名结合法：综合排名 = 评委排名 + 观众排名
    long_df["combined_rank"] = long_df["judge_rank"] + long_df["audience_rank"]

    # 排名结合法：最终排名（1为最好）
    long_df["combined_rank_final"] = long_df.groupby(["season", "week"])["combined_rank"].rank(
        ascending=True, method="min"
    )

    # 基于淘汰概率预测淘汰（所有赛季）
    print("- 根据淘汰概率预测淘汰...")
    def mark_top_n_eliminated_by_prob(group):
        """标记该周淘汰概率最高的N个人"""
        n = group["n_eliminated"].iloc[0]
        if n == 0:
            group["pred_eliminated"] = 0
            return group
        group["pred_eliminated"] = 0
        # 淘汰概率越高越危险
        top_prob_idx = group.nlargest(int(n), "elimination_prob").index
        group.loc[top_prob_idx, "pred_eliminated"] = 1
        return group

    long_df = long_df.groupby(["season", "week"]).apply(mark_top_n_eliminated_by_prob).reset_index(drop=True)

    # 基于存活概率的统一最终排名（越高越安全，排名越好）
    # 规则：每赛季每周按存活概率排序；若该周淘汰多人，则被淘汰者排名相同
    def assign_final_rank(group):
        n_elim = int(group["n_eliminated"].iloc[0])
        # 先按存活概率（越高越好）给出基础排名
        base_rank = group["survival_prob"].rank(ascending=False, method="min")
        group = group.copy()
        group["final_rank"] = base_rank
        if n_elim > 0:
            # 找到淘汰的选手（存活概率最低）
            elim_idx = group.nsmallest(n_elim, "survival_prob").index
            # 淘汰者统一排名为该周最差名次
            worst_rank = base_rank.max()
            group.loc[elim_idx, "final_rank"] = worst_rank
        return group

    long_df = long_df.groupby(["season", "week"], group_keys=False).apply(assign_final_rank)

    # 导出两种方法下的排名关系表（最终排名一致，来自淘汰概率）
    share_rank_cols = [
        "season", "week", "celebrity_name",
        "judge_rank", "audience_share_rank", "final_rank", "pred_eliminated", "survival_prob"
    ]
    rank_rank_cols = [
        "season", "week", "celebrity_name",
        "judge_rank", "audience_rank", "final_rank", "pred_eliminated", "survival_prob"
    ]
    share_rank_df = long_df[share_rank_cols].copy()
    rank_rank_df = long_df[rank_rank_cols].copy()

    # 按赛季-周排序，确保按季度每周划分
    share_rank_df = share_rank_df.sort_values(["season", "week", "final_rank", "judge_rank"]).reset_index(drop=True)
    rank_rank_df = rank_rank_df.sort_values(["season", "week", "final_rank", "judge_rank"]).reset_index(drop=True)

    share_rank_path = project_root / "data" / "percentage_method_rankings.csv"
    rank_rank_path = project_root / "data" / "ranking_method_rankings.csv"
    try:
        share_rank_df.to_csv(share_rank_path, index=False, encoding="utf-8-sig")
        rank_rank_df.to_csv(rank_rank_path, index=False, encoding="utf-8-sig")
        print(f"[已导出] 百分比结合法排名表: {share_rank_path}")
        print(f"[已导出] 排名结合法排名表: {rank_rank_path}")
    except PermissionError:
        print(f"[警告] 无法写入排名表（文件被占用）")
    except Exception as e:
        print(f"[错误] 导出排名表失败: {e}")

    print("[完成] 观众投票比例和排名已预测")
    # 8. 模型评估
    print("\n[步骤 8] 模型评估...")
    
    print("\n- 淘汰概率分布")
    print(f"平均淘汰概率: {long_df['elimination_prob'].mean():.4f}")
    print(f"淘汰概率范围: [{long_df['elimination_prob'].min():.4f}, {long_df['elimination_prob'].max():.4f}]")
    
    print("\n- 观众投票比例分布")
    print(f"平均投票比例: {long_df['audience_share'].mean():.4f}")
    print(f"投票比例范围: [{long_df['audience_share'].min():.4f}, {long_df['audience_share'].max():.4f}]")
    
    print("\n- 淘汰预测准确性")
    # 综合准确率（所有选手-周记录的准确率）
    elimination_accuracy = (long_df["pred_eliminated"] == long_df["is_eliminated"]).mean()
    print(f"记录级别准确率: {elimination_accuracy:.4f} (所有选手-周记录)")
    
    # 只针对被淘汰选手的准确率（召回率 Recall）
    eliminated_only = long_df[long_df["is_eliminated"] == 1]
    if len(eliminated_only) > 0:
        recall = (eliminated_only["pred_eliminated"] == 1).mean()
        print(f"淘汰者识别准确率 (Recall): {recall:.4f} (仅统计真正被淘汰的{len(eliminated_only)}人)")
    else:
        recall = None
        print(f"淘汰者识别准确率 (Recall): N/A (无淘汰记录)")
    
    # 每周平均准确率（每周单独计算准确率再平均）
    week_level_accuracy = long_df.groupby(["season", "week"]).apply(
        lambda g: (g["pred_eliminated"] == g["is_eliminated"]).mean()
    ).mean()
    print(f"周平均准确率: {week_level_accuracy:.4f} (每周准确率的平均)")
    
    # 周级别完全正确率（整周所有人都预测对才算对）
    week_perfect_accuracy = long_df.groupby(["season", "week"]).apply(
        lambda g: (g["pred_eliminated"] == g["is_eliminated"]).all()
    ).mean()
    print(f"周级别完全正确率: {week_perfect_accuracy:.4f} (整周所有人都对)")
    
    # 按淘汰人数类型分析（仅统计有效周）
    valid_mask = long_df["judge_total_score"] > 0
    valid_long_df = long_df[valid_mask]
    for n in [0, 1, 2, 3, 4]:
        subset = valid_long_df[valid_long_df["n_eliminated"] == n]
        if len(subset) > 0:
            acc = (subset["pred_eliminated"] == subset["is_eliminated"]).mean()
            n_weeks = subset.groupby(["season", "week"]).ngroups
            if n_weeks > 0:
                print(f"  - {n}人淘汰周 ({n_weeks}周): {acc:.4f}")

    '''
    
    # ========== 一致性评估 ==========
    print("\n=== 一致性评估：能否定位淘汰者 ===")
    
    # Hit@2
    hit_at_2 = calculate_hit_at_k(long_df, predicted_col="elimination_prob", k=2, higher_is_worse=True)
    if not pd.isna(hit_at_2.get('hit_rate')):
        print(f"Hit@2 命中率: {hit_at_2['hit_rate']:.4f} ({hit_at_2['hit_count']:.2f}/{hit_at_2['total_eliminations']})")
    
    # Hit@3
    hit_at_3 = calculate_hit_at_k(long_df, predicted_col="elimination_prob", k=3, higher_is_worse=True)
    if not pd.isna(hit_at_3.get('hit_rate')):
        print(f"Hit@3 命中率: {hit_at_3['hit_rate']:.4f} ({hit_at_3['hit_count']:.2f}/{hit_at_3['total_eliminations']})")

    # 按淘汰人数分组的 Hit@k
    def _print_hit_by_n(hit_result: dict, label: str):
        details = hit_result.get("details")
        if details is None or details.empty:
            return
        grouped = details.groupby("n_eliminated")[f"hit_at_{label}"]
        for n_elim, mean_hit in grouped.mean().items():
            print(f"  - {int(n_elim)}人淘汰周 Hit@{label}: {mean_hit:.4f}")

    print("- Hit@2 按淘汰人数分组：")
    _print_hit_by_n(hit_at_2, 2)

    print("- Hit@3 按淘汰人数分组：")
    _print_hit_by_n(hit_at_3, 3)
    
    # 淘汰边界间隔分布
    print("\n淘汰边界间隔分析:")
    
    margin = analyze_boundary_margin_distribution(long_df, predicted_col="elimination_prob", higher_is_worse=True, by_elimination_count=True)
    print(f"  全局统计:")
    print(f"    - 均值: {margin['mean']:.4f}")
    print(f"    - 标准差: {margin['std']:.4f}")
    print(f"    - 范围: [{margin['min']:.4f}, {margin['max']:.4f}]")
    print(f"    - 中位数: {margin['median']:.4f}")
    print(f"    - 四分位数范围: [{margin['q25']:.4f}, {margin['q75']:.4f}]")
    if 'by_elimination_count' in margin:
        print(f"  按淘汰人数分组:")
        for n_elim, stats in margin['by_elimination_count'].items():
            print(f"    {n_elim}人淘汰 ({stats['count']}周): 均值 {stats['mean']:.4f}, 标准差 {stats['std']:.4f}, 范围 [{stats['min']:.4f}, {stats['max']:.4f}]")
   
    # ========== 不确定性分析 ==========
    print("\n=== 不确定性分析：投票区间和方差 ===")
    
    print("- Bootstrap 分析（500次重采样）...")
    uncertainty = analyze_vote_share_intervals(
        long_df,
        feature_cols=feature_cols,
        model_class=EliminationProbModel,
        train_mask=share_train_mask,
        vote_share_col="audience_share",
        confidence=0.95,
        n_bootstrap=500
    )
    print(f"  选手级别分析:")
    print(f"    - 平均区间宽度 (95% CI): {uncertainty['mean_interval_width']:.4f}")
    print(f"    - 平均后验方差: {uncertainty['mean_variance']:.4f}")
    
    # 投票总数和个人票数的确定性分析
    print("\n- 投票总数确定性分析...")
    vote_count_uncertainty = analyze_vote_count_uncertainty(
        bootstrap_samples=uncertainty['bootstrap_samples'],
        long_df=long_df,
        base_total_votes=5e6,  # 基准 500万 票/周
        votes_variation_by_week=True,  # 允许每周不同
        random_state=42
    )
    
    overall_stats = vote_count_uncertainty['overall_stats']
    print(f"  投票总数设置:")
    print(f"    - 基准值: {overall_stats['base_total_votes']:.0f} 票/周")
    print(f"    - 平均值: {overall_stats['mean_weekly_votes']:.0f} 票/周")
    print(f"    - 标准差: {overall_stats['std_weekly_votes']:.0f} 票")
    print(f"  个人票数不确定性:")
    print(f"    - 平均变异系数(CV): {overall_stats['mean_individual_cv']:.4f}")
    print(f"    - 平均票数区间宽度(95% CI): {overall_stats['mean_individual_ci_width']:.0f} 票")
    print(f"    - 平均标准差: {overall_stats['mean_individual_std']:.0f} 票")
    print(f"    - 平均相对区间宽度: {overall_stats['mean_relative_ci_width']:.4f}")
    
    # 导出详细的选手级别不确定性数据
    individual_uncertainty = vote_count_uncertainty['individual_summary']
    output_path = project_root / 'data' / 'individual_vote_uncertainty.csv'
    try:
        individual_uncertainty.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  [已导出] 选手级别不确定性数据: {output_path}")
        print(f"    - 共 {len(individual_uncertainty)} 条记录")
        print(f"    - 列名: {list(individual_uncertainty.columns)}")
        print(f"\n  前5条示例:")
        for _, row in individual_uncertainty.head(5).iterrows():
            print(f"    S{int(row['season'])}W{int(row['week'])} {row['celebrity_name']}: "
                  f"票数={row['vote_mean']:.0f}, CI宽度={row['vote_ci_width']:.0f}, "
                  f"CV={row['coefficient_of_variation']:.4f}")
    except PermissionError:
        print(f"\n  [警告] 无法写入: {output_path}（文件被占用）")
    except Exception as e:
        print(f"\n  [错误] 导出失败: {e}")
    
    
    # ========== 公平性评估 ==========
    print("\n=== 公平性/偏向性评估 ===")
    
    long_df["judge_rank"] = long_df.groupby(["season", "week"])["judge_total_score"].rank(ascending=True, method="min")
    fairness = reversal_ratios(long_df, judge_rank_col="judge_rank", audience_rank_col="audience_rank_score")
    
    print(f"裁判与粉丝投票的逆转情况:")
    print(f"  - 裁判被粉丝逆转比例: {fairness['judge_overturned_ratio']:.4f}")
    print(f"    (裁判给低分但粉丝投票高 → 粉丝对评分的抗议)")
    print(f"  - 粉丝被裁判逆转比例: {fairness['audience_overturned_ratio']:.4f}")
    print(f"    (粉丝投票低但裁判给高分 → 裁判与粉丝偏好不同)")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    '''


if __name__ == "__main__":
    log_dir = project_root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"terminal_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = _TeeStdout(original_stdout, f)
        try:
            print(f"[日志] 终端输出将同步保存至: {log_path}")
            main()
            print(f"[日志] 输出保存完成: {log_path}")
        finally:
            sys.stdout = original_stdout
