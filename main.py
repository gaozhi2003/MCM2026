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
from evaluation.comprehensive_metrics import comprehensive_evaluation
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


def main(export_only_three_csvs=False, output_dir=None):
    """主程序
    
    Args:
        export_only_three_csvs: 若为 True，运行至三种方法排名表导出后即退出
        output_dir: 导出目录，None 时使用 data/
    """
    base_export_dir = Path(output_dir) if output_dir else project_root / 'data'
    print("=" * 60)
    title = "MCM 问题 C - 仅导出三种方法排名表" if export_only_three_csvs else "MCM 问题 C 数据分析模型"
    print(title)
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

    # 两种方法各自按“综合排名”决定淘汰与周内排名（用于任务二对比：百分比法 vs 排名法结果可不同）
    def mark_eliminated_by_combined_share(group):
        n_elim = int(group["n_eliminated"].iloc[0])
        group = group.copy()
        group["pred_eliminated_pct"] = 0
        if n_elim > 0:
            # 百分比法：combined_share_rank 越大越差，淘汰最差的 n_elim 人
            worst_idx = group.nlargest(int(n_elim), "combined_share_rank").index
            group.loc[worst_idx, "pred_eliminated_pct"] = 1
        return group

    def mark_eliminated_by_combined_rank(group):
        n_elim = int(group["n_eliminated"].iloc[0])
        group = group.copy()
        group["pred_eliminated_rank"] = 0
        if n_elim > 0:
            # 排名法：combined_rank_final 越大越差，淘汰最差的 n_elim 人
            worst_idx = group.nlargest(int(n_elim), "combined_rank_final").index
            group.loc[worst_idx, "pred_eliminated_rank"] = 1
        return group

    long_df = long_df.groupby(["season", "week"], group_keys=False).apply(mark_eliminated_by_combined_share).reset_index(drop=True)
    long_df = long_df.groupby(["season", "week"], group_keys=False).apply(mark_eliminated_by_combined_rank).reset_index(drop=True)

    # 基于淘汰概率预测淘汰（用于一致性等评估，保留原逻辑）
    print("- 根据淘汰概率预测淘汰...")
    def mark_top_n_eliminated_by_prob(group):
        """标记该周淘汰概率最高的N个人"""
        n = group["n_eliminated"].iloc[0]
        if n == 0:
            group["pred_eliminated"] = 0
            return group
        group["pred_eliminated"] = 0
        top_prob_idx = group.nlargest(int(n), "elimination_prob").index
        group.loc[top_prob_idx, "pred_eliminated"] = 1
        return group

    long_df = long_df.groupby(["season", "week"]).apply(mark_top_n_eliminated_by_prob).reset_index(drop=True)

    # 基于存活概率的统一最终排名（用于一致性等评估，保留原逻辑）
    def assign_final_rank(group):
        n_elim = int(group["n_eliminated"].iloc[0])
        base_rank = group["survival_prob"].rank(ascending=False, method="min")
        group = group.copy()
        group["final_rank"] = base_rank
        if n_elim > 0:
            elim_idx = group.nsmallest(n_elim, "survival_prob").index
            worst_rank = base_rank.max()
            group.loc[elim_idx, "final_rank"] = worst_rank
        return group

    long_df = long_df.groupby(["season", "week"], group_keys=False).apply(assign_final_rank)

    # 导出两种方法下的排名关系表（各用各自的综合排名与淘汰：百分比法用 combined_share_rank，排名法用 combined_rank_final）
    share_rank_df = long_df[[
        "season", "week", "celebrity_name",
        "judge_rank", "audience_share_rank",
        "combined_share_rank", "pred_eliminated_pct", "survival_prob"
    ]].copy()
    share_rank_df = share_rank_df.rename(columns={"combined_share_rank": "final_rank", "pred_eliminated_pct": "pred_eliminated"})

    rank_rank_df = long_df[[
        "season", "week", "celebrity_name",
        "judge_rank", "audience_rank",
        "combined_rank_final", "pred_eliminated_rank", "survival_prob"
    ]].copy()
    rank_rank_df = rank_rank_df.rename(columns={"combined_rank_final": "final_rank", "pred_eliminated_rank": "pred_eliminated"})

    # 按赛季-周排序，确保按季度每周划分
    share_rank_df = share_rank_df.sort_values(["season", "week", "final_rank", "judge_rank"]).reset_index(drop=True)
    rank_rank_df = rank_rank_df.sort_values(["season", "week", "final_rank", "judge_rank"]).reset_index(drop=True)

    share_rank_path = base_export_dir / "percentage_method_rankings.csv"
    rank_rank_path = base_export_dir / "ranking_method_rankings.csv"
    try:
        base_export_dir.mkdir(parents=True, exist_ok=True)
        share_rank_df.to_csv(share_rank_path, index=False, encoding="utf-8-sig")
        rank_rank_df.to_csv(rank_rank_path, index=False, encoding="utf-8-sig")
        print(f"[已导出] 百分比结合法排名表: {share_rank_path}")
        print(f"[已导出] 排名结合法排名表: {rank_rank_path}")
    except PermissionError:
        print(f"[警告] 无法写入排名表（文件被占用）")
    except Exception as e:
        print(f"[错误] 导出排名表失败: {e}")

    # 周级份额表：观众份额、评委份额、总体份额（供作图与“是否更偏观众”分析）
    weekly_shares_path = base_export_dir / "weekly_shares.csv"
    weekly_shares_df = long_df[
        ["season", "week", "celebrity_name", "judge_share", "audience_share", "combined_share"]
    ].sort_values(["season", "week", "celebrity_name"]).reset_index(drop=True)
    try:
        weekly_shares_df.to_csv(weekly_shares_path, index=False, encoding="utf-8-sig")
        print(f"[已导出] 周级份额表: {weekly_shares_path}")
    except Exception as e:
        print(f"[错误] 导出周级份额表失败: {e}")

    print("[完成] 观众投票比例和排名已预测")
    
    # ========== 评委与观众排名差异分析 ==========
    print("\n=== 评委与观众排名差异分析 ===")

    # 百分比结合法（观众排名 = audience_share_rank）
    long_df["rank_diff_share"] = abs(long_df["judge_rank"] - long_df["audience_share_rank"])
    long_df["rank_direction_share"] = long_df["judge_rank"] - long_df["audience_share_rank"]

    high_diff_share = long_df[long_df["rank_diff_share"] >= 2].copy()
    if len(high_diff_share) > 0:
        print(f"\n[百分比结合法] 排名差异>=2案例数: {len(high_diff_share)}")
        case1_share = high_diff_share[high_diff_share["rank_direction_share"] > 0]
        case2_share = high_diff_share[high_diff_share["rank_direction_share"] < 0]

        print(f"  第1类(评委低/观众高): {len(case1_share)}人次")
        print(f"  第2类(评委高/观众低): {len(case2_share)}人次")

        share_comp = high_diff_share[[
            "season", "week", "celebrity_name",
            "judge_rank", "audience_share_rank", "judge_share", "audience_share",
            "rank_diff_share", "rank_direction_share", "survival_prob"
        ]].sort_values(["rank_direction_share", "rank_diff_share"], ascending=[False, False]).reset_index(drop=True)

        share_comp_path = project_root / "data" / "ranking_discrepancy_analysis_percentage.csv"
        try:
            share_comp.to_csv(share_comp_path, index=False, encoding="utf-8-sig")
            print(f"  [已导出] 百分比结合法差异表: {share_comp_path}")
        except Exception as e:
            print(f"  [错误] 导出失败: {e}")
    else:
        print("[百分比结合法] 无排名差异>=2的案例")

    # 排名结合法（观众排名 = audience_rank）
    long_df["rank_diff_rank"] = abs(long_df["judge_rank"] - long_df["audience_rank"])
    long_df["rank_direction_rank"] = long_df["judge_rank"] - long_df["audience_rank"]

    high_diff_rank = long_df[long_df["rank_diff_rank"] >= 2].copy()
    if len(high_diff_rank) > 0:
        print(f"\n[排名结合法] 排名差异>=2案例数: {len(high_diff_rank)}")
        case1_rank = high_diff_rank[high_diff_rank["rank_direction_rank"] > 0]
        case2_rank = high_diff_rank[high_diff_rank["rank_direction_rank"] < 0]

        print(f"  第1类(评委低/观众高): {len(case1_rank)}人次")
        print(f"  第2类(评委高/观众低): {len(case2_rank)}人次")

        rank_comp = high_diff_rank[[
            "season", "week", "celebrity_name",
            "judge_rank", "audience_rank", "rank_diff_rank", "rank_direction_rank", "survival_prob"
        ]].sort_values(["rank_direction_rank", "rank_diff_rank"], ascending=[False, False]).reset_index(drop=True)

        rank_comp_path = project_root / "data" / "ranking_discrepancy_analysis_rank.csv"
        try:
            rank_comp.to_csv(rank_comp_path, index=False, encoding="utf-8-sig")
            print(f"  [已导出] 排名结合法差异表: {rank_comp_path}")
        except Exception as e:
            print(f"  [错误] 导出失败: {e}")
    else:
        print("[排名结合法] 无排名差异>=2的案例")

    # ========== 新方法：末尾两位 + 评委最低 ==========
    print("\n=== 新方法：末尾两位 + 评委最低 ===")

    def mark_eliminated_by_bottom2_then_judge(group):
        """观众排名+评委排名相加，取最大的候选，再按评委评分淘汰最低者
        
        多淘汰周处理：候选人数量为 max(2, n_elim)
        """
        n_elim = int(group["n_eliminated"].iloc[0])
        group = group.copy()
        group["pred_eliminated_alt"] = 0
        
        # 综合排名分数 = 观众排名 + 评委排名（值越大越靠后）
        group["combined_rank_score"] = group["audience_rank"] + group["judge_rank"]
        
        # 初始化排名（暂时设为None，最后只设置被淘汰者）
        group["final_rank_alt"] = group["combined_rank_score"].rank(ascending=True, method="min")
        
        if n_elim == 0:
            return group
        
        # 取综合排名分数最大的 max(2, n_elim) 位作为候选人
        # 确保候选人数量至少为2，且不少于需要淘汰的人数
        n_candidates = max(2, n_elim)
        n_candidates = min(n_candidates, len(group))  # 不能超过总人数
        candidate_idx = group.nlargest(n_candidates, "combined_rank_score").index
        candidates = group.loc[candidate_idx]

        # 从候选人中按评委评分最低者淘汰（若要淘汰多人，则取最低的 n_elim 人）
        elim_idx = candidates.nsmallest(n_elim, "judge_total_score").index
        group.loc[elim_idx, "pred_eliminated_alt"] = 1
        
        # 被淘汰者排名设为该周最差名次（即总人数）
        worst_rank = len(group)
        group.loc[elim_idx, "final_rank_alt"] = worst_rank
        
        return group

    long_df = long_df.groupby(["season", "week"], group_keys=False).apply(mark_eliminated_by_bottom2_then_judge)

    # 新方法整体准确率
    alt_accuracy = (long_df["pred_eliminated_alt"] == long_df["is_eliminated"]).mean()
    print(f"新方法记录级别准确率: {alt_accuracy:.4f}")

    # 新方法淘汰者识别准确率（召回率）
    alt_eliminated_only = long_df[long_df["is_eliminated"] == 1]
    if len(alt_eliminated_only) > 0:
        alt_recall = (alt_eliminated_only["pred_eliminated_alt"] == 1).mean()
        print(f"新方法淘汰者识别准确率(Recall): {alt_recall:.4f} (仅统计真正被淘汰的{len(alt_eliminated_only)}人)")
    else:
        alt_recall = None
        print("新方法淘汰者识别准确率(Recall): N/A (无淘汰记录)")

    # 按差异大小分析（排名结合法差异）
    diff_threshold = 2
    long_df["rank_diff_rank"] = abs(long_df["judge_rank"] - long_df["audience_rank"])
    small_diff = long_df[long_df["rank_diff_rank"] < diff_threshold]
    large_diff = long_df[long_df["rank_diff_rank"] >= diff_threshold]
    if len(small_diff) > 0:
        acc_small = (small_diff["pred_eliminated_alt"] == small_diff["is_eliminated"]).mean()
        print(f"小差异(<{diff_threshold})样本数 {len(small_diff)}，新方法准确率: {acc_small:.4f}")
    if len(large_diff) > 0:
        acc_large = (large_diff["pred_eliminated_alt"] == large_diff["is_eliminated"]).mean()
        print(f"大差异(>={diff_threshold})样本数 {len(large_diff)}，新方法准确率: {acc_large:.4f}")

    # 两类案例分析（排名结合法）
    case1_rank = long_df[long_df["judge_rank"] > long_df["audience_rank"]]
    case2_rank = long_df[long_df["judge_rank"] < long_df["audience_rank"]]
    if len(case1_rank) > 0:
        acc_case1 = (case1_rank["pred_eliminated_alt"] == case1_rank["is_eliminated"]).mean()
        print(f"第1类(评委低/观众高)样本数 {len(case1_rank)}，新方法准确率: {acc_case1:.4f}")
    if len(case2_rank) > 0:
        acc_case2 = (case2_rank["pred_eliminated_alt"] == case2_rank["is_eliminated"]).mean()
        print(f"第2类(评委高/观众低)样本数 {len(case2_rank)}，新方法准确率: {acc_case2:.4f}")

    # 导出新方法对比表
    alt_cols = [
        "season", "week", "celebrity_name",
        "judge_rank", "audience_rank", "judge_total_score",
        "rank_diff_rank", "pred_eliminated_alt", "is_eliminated"
    ]
    alt_df = long_df[alt_cols].sort_values(["season", "week", "pred_eliminated_alt", "judge_rank"], ascending=[True, True, False, True]).reset_index(drop=True)
    alt_path = project_root / "data" / "elimination_rule_bottom2_then_judge.csv"
    try:
        alt_df.to_csv(alt_path, index=False, encoding="utf-8-sig")
        print(f"[已导出] 新方法结果表: {alt_path}")
    except Exception as e:
        print(f"[错误] 导出新方法结果表失败: {e}")
    
    # 导出新方法排名表（与percentage_method_rankings.csv格式对应）
    new_rank_cols = [
        "season", "week", "celebrity_name",
        "judge_rank", "audience_rank", "final_rank_alt", "pred_eliminated_alt", "judge_total_score"
    ]
    new_rank_df = long_df[new_rank_cols].copy()
    new_rank_df = new_rank_df.sort_values(["season", "week", "final_rank_alt", "judge_rank"]).reset_index(drop=True)
    new_rank_path = base_export_dir / "new_method_rankings.csv"
    try:
        new_rank_df.to_csv(new_rank_path, index=False, encoding="utf-8-sig")
        print(f"[已导出] 新方法排名表: {new_rank_path}")
    except Exception as e:
        print(f"[错误] 导出新方法排名表失败: {e}")

    if export_only_three_csvs:
        print("\n[完成] 三种方法排名表已导出，已退出（跳过后续评估等步骤）")
        return

    # ========== 三类指标综合评估 ==========
    print("\n" + "=" * 70)
    print("【综合评估】新旧方法对比 - 稳定性、抗操纵性、一致性")
    print("=" * 70)
    
    metrics_result = comprehensive_evaluation(long_df, output_dir=str(project_root / 'data'))
    
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
    
    # 详细输出每个赛季每周的Δ、Hit@2、Hit@3
    print("\n- 每个赛季每周的详细指标（前30周示例）：")
    print("注：Δ（边界间隔）= 第n名（淘汰最后一人）的分数 - 第n+1名（安全者）的分数")
    
    # 获取Hit@k的详细数据
    hit2_details = hit_at_2.get('details')
    hit3_details = hit_at_3.get('details')
    
    if hit2_details is not None and not hit2_details.empty:
        # 为Hit@2和Hit@3分别计算对应的边界间隔
        # 边界间隔基于实际淘汰人数n来计算：第n名与第n+1名的差距
        weekly_metrics_data = []
        for (season, week), group in long_df.groupby(["season", "week"]):
            n_elim = int(group["is_eliminated"].sum())
            if n_elim == 0:
                continue
            
            # 按淘汰概率排序（高分=更危险）
            scores_series = group.set_index("celebrity_name")["elimination_prob"].sort_values(ascending=False)
            
            # 边界间隔Δ：基于实际淘汰人数n
            # Δ = 第n名（淘汰最后一人）的分数 - 第n+1名（安全者）的分数
            boundary_margin = np.nan
            if len(scores_series) > n_elim:
                eliminated_last_score = scores_series.iloc[n_elim - 1]  # 第n名
                safe_first_score = scores_series.iloc[n_elim]  # 第n+1名
                boundary_margin = eliminated_last_score - safe_first_score
            
            # 获取Hit@2和Hit@3的值
            hit2_val = hit2_details[(hit2_details['season'] == season) & (hit2_details['week'] == week)]['hit_at_2'].values
            hit2_val = hit2_val[0] if len(hit2_val) > 0 else np.nan
            
            hit3_val = hit3_details[(hit3_details['season'] == season) & (hit3_details['week'] == week)]['hit_at_3'].values
            hit3_val = hit3_val[0] if len(hit3_val) > 0 else np.nan
            
            weekly_metrics_data.append({
                'season': season,
                'week': week,
                'n_eliminated': n_elim,
                'hit_at_2': hit2_val,
                'hit_at_3': hit3_val,
                'boundary_margin': boundary_margin
            })
        
        weekly_metrics = pd.DataFrame(weekly_metrics_data)
        
        # 按赛季和周排序
        weekly_metrics = weekly_metrics.sort_values(['season', 'week']).reset_index(drop=True)
        
        # 打印表头
        print(f"\n{'Season':<8} {'Week':<6} {'N':<4} {'Hit@2':<8} {'Δ_2':<12} {'Hit@3':<8} {'Δ_3':<12}")
        print("-" * 80)
        
        # 打印前30周数据
        for idx, row in weekly_metrics.head(30).iterrows():
            season = int(row['season'])
            week = int(row['week'])
            n_elim = int(row['n_eliminated'])
            hit2 = row['hit_at_2']
            hit3 = row['hit_at_3']
            margin = row['boundary_margin']
            
            hit2_str = f"{hit2:.4f}" if not pd.isna(hit2) else "N/A"
            hit3_str = f"{hit3:.4f}" if not pd.isna(hit3) else "N/A"
            margin_str = f"{margin:.6f}" if not pd.isna(margin) else "N/A"
            
            print(f"S{season:<7} W{week:<5} {n_elim:<4} {hit2_str:<8} {hit3_str:<8} {margin_str:<12}")
        
        if len(weekly_metrics) > 30:
            print(f"... (共{len(weekly_metrics)}周，仅显示前30周)")
        
        # 导出完整数据
        weekly_metrics_path = project_root / "data" / "weekly_hit_and_margin_metrics.csv"
        try:
            weekly_metrics.to_csv(weekly_metrics_path, index=False, encoding='utf-8-sig')
            print(f"\n  [已导出] 完整周级别指标: {weekly_metrics_path}")
        except Exception as e:
            print(f"\n  [警告] 导出失败: {e}")
    
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
   
    '''
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
    import sys
    export_only = "--export-three-csvs-only" in sys.argv
    out_dir = None
    for i, a in enumerate(sys.argv):
        if a == "--output-dir" and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
            break
    main(export_only_three_csvs=export_only, output_dir=out_dir)
