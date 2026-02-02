"""
四种方法的综合指标评估
1. 稳定性（Bootstrap翻转率）
2. 抗操纵性（攻击模拟）
3. 一致性（Spearman相关系数）

四种方法：
- 百分比结合法：combined_share 最低的 n 个人淘汰
- 排名结合法：combined_rank 最高的 n 个人淘汰
- 新方法（末尾两位+评委最低）：combined_rank_score 最高的2位候选，从中选评委评分最低的淘汰
- 动态权重法：S型曲线动态权重计算综合分数，候选后按评委评分淘汰
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
from evaluation.adaptive_weight_method import apply_adaptive_weight_method
warnings.filterwarnings('ignore')


def _predict_percentage_method(long_df):
    """百分比结合法：按 combined_share（评委+观众百分比）最低的n个淘汰"""
    preds = pd.Series(0, index=long_df.index, dtype=int)
    for (season, week), group in long_df.groupby(['season', 'week']):
        if 'n_eliminated' in group.columns:
            n_elim = int(group['n_eliminated'].iloc[0])
        else:
            n_elim = int(group['is_eliminated'].sum())
        if n_elim <= 0:
            continue
        # combined_share 越低越危险（分数低=排名差）
        bottom_idx = group.nsmallest(n_elim, 'combined_share').index
        preds.loc[bottom_idx] = 1
    return preds


def _predict_rank_method(long_df):
    """排名结合法：按 combined_rank（评委排名+观众排名）最高的n个淘汰"""
    preds = pd.Series(0, index=long_df.index, dtype=int)
    for (season, week), group in long_df.groupby(['season', 'week']):
        if 'n_eliminated' in group.columns:
            n_elim = int(group['n_eliminated'].iloc[0])
        else:
            n_elim = int(group['is_eliminated'].sum())
        if n_elim <= 0:
            continue
        # combined_rank 越高越危险（排名和大=综合排名差）
        top_idx = group.nlargest(n_elim, 'combined_rank').index
        preds.loc[top_idx] = 1
    return preds





def _predict_new_from_scores_with_audience_rank(long_df):
    """新方法：使用已有的 audience_rank 和 judge_rank 进行预测
    
    多淘汰周处理：如果 n_elim > 2，则候选人数量扩展为 max(2, n_elim)，
    确保有足够的候选人可供选择。
    """
    preds = pd.Series(0, index=long_df.index, dtype=int)
    for (season, week), group in long_df.groupby(['season', 'week']):
        if 'n_eliminated' in group.columns:
            n_elim = int(group['n_eliminated'].iloc[0])
        else:
            n_elim = int(group['is_eliminated'].sum())
        if n_elim <= 0:
            continue
        
        # 使用已有的 audience_rank 和 judge_rank
        audience_rank = group['audience_rank']
        judge_rank = group['judge_rank']
        
        # 综合排名分数 = 观众排名 + 评委排名（值越大越靠后）
        combined_rank_score = audience_rank + judge_rank
        
        # 取综合排名分数最大的 max(2, n_elim) 位作为候选人
        # 确保候选人数量至少为2，且不少于需要淘汰的人数
        n_candidates = max(2, n_elim)
        n_candidates = min(n_candidates, len(group))  # 不能超过总人数
        candidate_idx = combined_rank_score.nlargest(n_candidates).index
        candidates = group.loc[candidate_idx]

        # 从候选人中按评委评分淘汰最低的 n_elim 个
        elim_idx = candidates.nsmallest(n_elim, 'judge_total_score').index
        preds.loc[elim_idx] = 1
    return preds


def _predict_new_from_scores_with_audience_share(long_df):
    """百分比结合法 + Bottom 2 + 评委裁决：combined_share 最低的 max(2,n_elim) 位作为候选人，评委在候选人中淘汰得分最低者"""
    preds = pd.Series(0, index=long_df.index, dtype=int)
    for (season, week), group in long_df.groupby(['season', 'week']):
        n_elim = int(group['is_eliminated'].sum())
        if n_elim <= 0:
            continue
        if 'combined_share' not in group.columns:
            continue
        n_candidates = max(2, n_elim)
        n_candidates = min(n_candidates, len(group))
        candidate_idx = group.nsmallest(n_candidates, 'combined_share').index
        candidates = group.loc[candidate_idx]
        elim_idx = candidates.nsmallest(n_elim, 'judge_total_score').index
        preds.loc[elim_idx] = 1
    return preds


def bootstrap_flip_rate(long_df, method='percentage', n_bootstrap=50, noise_level=0.05, random_state=42):
    """计算稳定性指标：Bootstrap翻转率（对观众投票占比 audience_share 添加随机噪声）"""
    rng = np.random.default_rng(random_state)

    # 必须有 audience_share 列
    if 'audience_share' not in long_df.columns:
        return np.nan

    # 获取基准预测
    if method == 'percentage':
        base_pred = _predict_percentage_method(long_df).values
    elif method == 'rank':
        base_pred = _predict_rank_method(long_df).values
    elif method == 'new':
        base_pred = _predict_new_from_scores_with_audience_rank(long_df).values
    elif method == 'new_pct':
        base_pred = _predict_new_from_scores_with_audience_share(long_df).values
    else:
        return np.nan
    
    total_flips = 0
    for _ in range(n_bootstrap):
        noisy_df = long_df.copy()
        # 对每个人的 audience_share 添加噪声
        for (season, week), group in noisy_df.groupby(['season', 'week']):
            week_mask = (noisy_df['season'] == season) & (noisy_df['week'] == week)
            week_data = noisy_df.loc[week_mask, 'audience_share'].values
            
            # 添加随机噪声（相对噪声，避免负值）
            noise = rng.normal(0, noise_level, len(week_data))
            noisy_shares = week_data * (1 + noise)
            noisy_shares = np.clip(noisy_shares, 0, None)  # 确保非负
            
            # 重新归一化（周内和为1）
            noisy_shares = noisy_shares / noisy_shares.sum()
            noisy_df.loc[week_mask, 'audience_share'] = noisy_shares
        
        # 重新计算 combined_share（评委百分比 + 观众百分比）
        noisy_df['combined_share'] = noisy_df['judge_share'] + noisy_df['audience_share']
        
        # 重新计算观众排名
        noisy_df['audience_rank'] = noisy_df.groupby(['season', 'week'])['audience_share'].rank(
            ascending=False, method='min'
        )
        
        # 重新计算 combined_rank（评委排名 + 观众排名）
        noisy_df['combined_rank'] = noisy_df['judge_rank'] + noisy_df['audience_rank']
        
        # 根据方法重新预测
        if method == 'percentage':
            noisy_pred = _predict_percentage_method(noisy_df).values
        elif method == 'rank':
            noisy_pred = _predict_rank_method(noisy_df).values
        elif method == 'new':
            noisy_pred = _predict_new_from_scores_with_audience_rank(noisy_df).values
        elif method == 'new_pct':
            noisy_pred = _predict_new_from_scores_with_audience_share(noisy_df).values
        else:
            noisy_pred = base_pred
        
        total_flips += (base_pred != noisy_pred).sum()
    return total_flips / (len(base_pred) * n_bootstrap)


def attack_robustness(long_df, method='percentage', n_attacks=100, attack_strength=0.5, random_state=4):
    """计算抗操纵性指标：攻击下的结果偏移率（攻击观众投票占比 audience_share）"""
    rng = np.random.default_rng(random_state)

    # 必须有 audience_share 列
    if 'audience_share' not in long_df.columns:
        return np.nan

    # 获取基准预测
    if method == 'percentage':
        base_pred = _predict_percentage_method(long_df).values
    elif method == 'rank':
        base_pred = _predict_rank_method(long_df).values
    elif method == 'new':
        base_pred = _predict_new_from_scores_with_audience_rank(long_df).values
    elif method == 'new_pct':
        base_pred = _predict_new_from_scores_with_audience_share(long_df).values
    else:
        return np.nan
    
    base_df = long_df.copy()
    change_rates = []
    for _ in range(n_attacks):
        attacked = base_df.copy()
        # 随机选择一个人，提升其 audience_share
        attack_idx = rng.integers(0, len(attacked))
        attacked_row = attacked.iloc[attack_idx]
        season, week = attacked_row['season'], attacked_row['week']
        
        # 获取该周的所有人
        week_mask = (attacked['season'] == season) & (attacked['week'] == week)
        week_data = attacked[week_mask].copy()
        
        # 提升被攻击者的 audience_share
        attack_within_week = week_data.index.get_loc(attacked.index[attack_idx])
        week_data.iloc[attack_within_week, week_data.columns.get_loc('audience_share')] *= (1 + attack_strength)
        
        # 重新归一化该周的 audience_share（因为周内和应该为1）
        total_share = week_data['audience_share'].sum()
        week_data['audience_share'] = week_data['audience_share'] / total_share
        
        # 更新到attacked数据中
        attacked.loc[week_mask, 'audience_share'] = week_data['audience_share'].values
        
        # 重新计算 combined_share
        attacked['combined_share'] = attacked['judge_share'] + attacked['audience_share']
        
        # 重新计算观众排名
        attacked['audience_rank'] = attacked.groupby(['season', 'week'])['audience_share'].rank(
            ascending=False, method='min'
        )
        
        # 重新计算 combined_rank
        attacked['combined_rank'] = attacked['judge_rank'] + attacked['audience_rank']
        
        # 根据方法重新预测
        if method == 'percentage':
            attacked_pred = _predict_percentage_method(attacked).values
        elif method == 'rank':
            attacked_pred = _predict_rank_method(attacked).values
        elif method == 'new':
            attacked_pred = _predict_new_from_scores_with_audience_rank(attacked).values
        elif method == 'new_pct':
            attacked_pred = _predict_new_from_scores_with_audience_share(attacked).values
        else:
            attacked_pred = base_pred
        
        change_rates.append((attacked_pred != base_pred).mean())
    return float(np.mean(change_rates))


def consistency_analysis(long_df, method='percentage'):
    """计算一致性指标（按周）：每周预测排名与实际 placement 的 Spearman，各周取平均"""
    if 'placement' not in long_df.columns:
        return 0

    # 根据方法选择排名列
    if method == 'percentage':
        if 'combined_share_rank' not in long_df.columns:
            return 0
        rank_col = 'combined_share_rank'
    elif method == 'rank':
        if 'combined_rank_final' not in long_df.columns:
            return 0
        rank_col = 'combined_rank_final'
    elif method == 'new':
        if 'final_rank_alt' not in long_df.columns:
            return 0
        rank_col = 'final_rank_alt'
    elif method == 'new_pct':
        if 'final_rank_alt_pct' not in long_df.columns:
            return 0
        rank_col = 'final_rank_alt_pct'
    else:
        return 0

    data = long_df[['placement', rank_col, 'season', 'week']].copy().dropna()
    spearman_scores = []

    for (season, week), group in data.groupby(['season', 'week']):
        if len(group) > 2:
            actual_rank = group['placement'].values
            pred_rank = group[rank_col].values
            try:
                corr, _ = stats.spearmanr(actual_rank, pred_rank)
                if not np.isnan(corr):
                    spearman_scores.append(corr)
            except Exception:
                continue

    return float(np.mean(spearman_scores)) if spearman_scores else 0


def _ensure_exit_week(long_df):
    """若缺少 exit_week 或全为 NaN，从 is_eliminated 推导：exit_week = 被淘汰的周，冠军用 max_week"""
    has_valid = 'exit_week' in long_df.columns and long_df['exit_week'].notna().any()
    if has_valid:
        return long_df
    if 'is_eliminated' not in long_df.columns:
        return long_df
    df = long_df.copy()
    df = df.drop(columns=['exit_week'], errors='ignore')
    elim = df[df['is_eliminated'] == 1][['season', 'celebrity_name', 'week']].drop_duplicates()
    elim = elim.rename(columns={'week': 'exit_week'})
    df = df.merge(elim, on=['season', 'celebrity_name'], how='left')
    mw = df.groupby('season')['week'].max().reset_index(name='max_week')
    df = df.merge(mw, on='season', how='left')
    df['exit_week'] = df['exit_week'].fillna(0).astype(int)
    df.loc[df['exit_week'] == 0, 'exit_week'] = df.loc[df['exit_week'] == 0, 'max_week'].astype(int)
    return df


def _compute_pred_final_rank_by_survival(long_df):
    """按每人最后出现的 week + 同周得分计算预测最终排名（无并列）。

    规则：
    - 最后存活的 week 越大 → 排名越好（数字越小）
    - 同周淘汰/退出的按当周 judge_total_score 排序，得分高排名好
    - celebrity_name 作为最终 tiebreaker 保证无并列

    返回：每人每季一行，含 pred_final_rank 列
    """
    need = ['placement', 'season', 'week', 'celebrity_name', 'judge_total_score']
    if not all(c in long_df.columns for c in need):
        return None

    df = _ensure_exit_week(long_df.copy())
    if 'exit_week' not in df.columns:
        return None
    if 'max_week' not in df.columns:
        max_week_df = df.groupby('season')['week'].max().reset_index(name='max_week')
        df = df.merge(max_week_df, on='season', how='left')
    df['last_week'] = df.apply(
        lambda r: int(r['exit_week']) if r['exit_week'] > 0 else int(r['max_week']),
        axis=1
    )

    # 每人每季只保留最后一周
    last = df[df['week'] == df['last_week']].drop_duplicates(
        subset=['season', 'celebrity_name'], keep='first'
    ).copy()

    # 按赛季：last_week 降序（存活越久越好），judge_total_score 降序（同周得分高越好），celebrity_name 稳定无并列
    pred_ranks = []
    for season, grp in last.groupby('season'):
        grp = grp.sort_values(
            by=['last_week', 'judge_total_score', 'celebrity_name'],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        grp['pred_final_rank'] = np.arange(1, len(grp) + 1, dtype=int)
        pred_ranks.append(grp[['season', 'celebrity_name', 'placement', 'pred_final_rank']])

    if not pred_ranks:
        return None
    return pd.concat(pred_ranks, ignore_index=True)


def _compute_all_methods_final_ranks(long_df):
    """计算三种方法各自的预测最终排名 + 存活+得分规则，返回合并表。

    每人取最后一周，按方法规则排序赋 rank：
    - survival: last_week desc, judge_total_score desc
    - percentage: last_week desc, combined_share_rank asc (1=best)
    - rank: last_week desc, combined_rank_final asc
    - new: last_week desc, final_rank_alt asc
    """
    survival_df = _compute_pred_final_rank_by_survival(long_df)
    if survival_df is None:
        return None

    df = long_df.copy()
    df = _ensure_exit_week(df)
    if 'exit_week' not in df.columns:
        return None
    if 'max_week' not in df.columns:
        df = df.merge(df.groupby('season')['week'].max().reset_index(name='max_week'), on='season', how='left')
    df['last_week'] = df.apply(
        lambda r: int(r['exit_week']) if r['exit_week'] > 0 else int(r['max_week']),
        axis=1
    )
    last = df[df['week'] == df['last_week']].drop_duplicates(subset=['season', 'celebrity_name'], keep='first').copy()

    result = survival_df[['season', 'celebrity_name', 'placement', 'pred_final_rank']].copy()
    result = result.rename(columns={'pred_final_rank': 'pred_final_rank_survival'})

    for out_col, sort_col in [
        ('pred_final_rank_pct', 'combined_share_rank'),
        ('pred_final_rank_rank', 'combined_rank_final'),
        ('pred_final_rank_new', 'final_rank_alt'),
        ('pred_final_rank_new_pct', 'final_rank_alt_pct'),
    ]:
        if sort_col not in last.columns:
            result[out_col] = np.nan
            continue
        ranks_list = []
        for season, grp in last.groupby('season'):
            grp = grp.sort_values(
                by=['last_week', sort_col, 'celebrity_name'],
                ascending=[False, True, True]
            ).reset_index(drop=True)
            r = grp[['season', 'celebrity_name']].copy()
            r[out_col] = np.arange(1, len(r) + 1, dtype=int)
            ranks_list.append(r)
        rank_df = pd.concat(ranks_list, ignore_index=True)
        result = result.merge(rank_df, on=['season', 'celebrity_name'], how='left')

    return result


def consistency_val_from_table(rank_table, pred_col):
    """从最终排名表计算某列与 placement 的 Spearman，按季取平均"""
    if rank_table is None or pred_col not in rank_table.columns:
        return np.nan
    data = rank_table[['placement', pred_col, 'season']].dropna()
    if len(data) == 0:
        return np.nan
    scores = []
    for season, grp in data.groupby('season'):
        if len(grp) > 2:
            try:
                c, _ = stats.spearmanr(grp['placement'].values, grp[pred_col].values)
                if not np.isnan(c):
                    scores.append(c)
            except Exception:
                pass
    return float(np.mean(scores)) if scores else np.nan


def consistency_final_rank(long_df, method=None):
    """计算最终排名一致性：预测最终排名与 placement 的 Spearman 相关系数。

    预测最终排名规则：
    - 按每人最后出现的 week：last_week 越大（存活越久）排名越好
    - 同周淘汰按当周 judge_total_score 排序，得分高排名好
    - 无并列

    method 参数保留兼容，本函数不再区分方法，统一使用上述规则。
    """
    data = _compute_pred_final_rank_by_survival(long_df)
    if data is None or len(data) == 0:
        return 0.0

    spearman_scores = []
    for season, group in data.groupby('season'):
        if len(group) > 2:
            try:
                corr, _ = stats.spearmanr(group['placement'].values, group['pred_final_rank'].values)
                if not np.isnan(corr):
                    spearman_scores.append(corr)
            except Exception:
                continue

    return float(np.mean(spearman_scores)) if spearman_scores else 0.0


def comprehensive_evaluation(long_df, output_dir='data'):
    """综合评估四种方法"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("【综合指标评估】四种方法对比 - 稳定性、抗操纵性、一致性")
    print("=" * 80)
    
    # 1. 稳定性
    print("\n【1】稳定性 (Stability) - Bootstrap翻转率")
    print("-" * 80)
    print("   定义: 投票数小幅扰动(±5%)下，淘汰判定翻转的概率")
    print("   越低越好（表示判定更稳定）\n")
    
    percentage_flip = bootstrap_flip_rate(long_df, method='percentage', n_bootstrap=50, noise_level=0.05)
    rank_flip = bootstrap_flip_rate(long_df, method='rank', n_bootstrap=50, noise_level=0.05)
    new_flip = bootstrap_flip_rate(long_df, method='new', n_bootstrap=50, noise_level=0.05)
    new_pct_flip = bootstrap_flip_rate(long_df, method='new_pct', n_bootstrap=50, noise_level=0.05)
    
    print(f"  百分比结合法翻转率:     {percentage_flip:.6f}")
    print(f"  排名结合法翻转率:     {rank_flip:.6f}")
    print(f"  排名+Bottom2翻转率:   {new_flip:.6f}")
    print(f"  份额+Bottom2翻转率:   {new_pct_flip:.6f}")
    
    methods_stability = [
        ('百分比结合法', percentage_flip), ('排名结合法', rank_flip),
        ('排名+Bottom2', new_flip), ('份额+Bottom2', new_pct_flip)
    ]
    stability_winner = min(methods_stability, key=lambda x: x[1])[0]
>>>>>>> 69fc835 (第二问的四种方法)
    print(f"  → {stability_winner} 更稳定（翻转率最低）")
    
    # 2. 抗操纵性
    print("\n【2】抗操纵性 (Robustness) - 单次攻击下的偏移率")
    print("-" * 80)
    print("   定义: 投票被操纵时，预测结果的变化幅度")
    print("   越低越好（表示抗操纵能力越强）\n")
    
    percentage_robust = attack_robustness(long_df, method='percentage', n_attacks=100)
    rank_robust = attack_robustness(long_df, method='rank', n_attacks=100)
    new_robust = attack_robustness(long_df, method='new', n_attacks=100)
    new_pct_robust = attack_robustness(long_df, method='new_pct', n_attacks=100)
    
    print(f"  百分比结合法结果偏移率:     {percentage_robust:.6f}")
    print(f"  排名结合法结果偏移率:     {rank_robust:.6f}")
    print(f"  排名+Bottom2结果偏移率:   {new_robust:.6f}")
    print(f"  份额+Bottom2结果偏移率:   {new_pct_robust:.6f}")
    
    methods_robust = [
        ('百分比结合法', percentage_robust), ('排名结合法', rank_robust),
        ('排名+Bottom2', new_robust), ('份额+Bottom2', new_pct_robust)
    ]
    robust_winner = min(methods_robust, key=lambda x: x[1])[0]
>>>>>>> 69fc835 (第二问的四种方法)
    print(f"  → {robust_winner} 更抗操纵（偏移率最低）")
    
    # 3. 一致性（最终排名）
    print("\n【3】一致性 (Consistency) - 预测最终排名与实际 placement 的 Spearman 相关系数")
    print("-" * 80)
    print("   每人取最后一周，按 last_week 降序 + 方法内 tiebreaker 赋预测最终排名")
    print("   按赛季计算 Spearman，各季取平均；越高越好\n")

    rank_table = _compute_all_methods_final_ranks(long_df)
    consistency_survival = consistency_final_rank(long_df)
    consistency_pct = consistency_val_from_table(rank_table, 'pred_final_rank_pct') if rank_table is not None else np.nan
    consistency_rank = consistency_val_from_table(rank_table, 'pred_final_rank_rank') if rank_table is not None else np.nan
    consistency_new = consistency_val_from_table(rank_table, 'pred_final_rank_new') if rank_table is not None else np.nan
    consistency_new_pct = consistency_val_from_table(rank_table, 'pred_final_rank_new_pct') if rank_table is not None else np.nan

    print(f"  存活+得分规则 Spearman: {consistency_survival:.4f}")
    print(f"  百分比结合法 Spearman:  {consistency_pct:.4f}" if not np.isnan(consistency_pct) else "  百分比结合法: N/A (缺少 combined_share_rank)")
    print(f"  排名结合法 Spearman:    {consistency_rank:.4f}" if not np.isnan(consistency_rank) else "  排名结合法: N/A (缺少 combined_rank_final)")
    print(f"  排名+Bottom2 Spearman:  {consistency_new:.4f}" if not np.isnan(consistency_new) else "  排名+Bottom2: N/A (缺少 final_rank_alt)")
    print(f"  份额+Bottom2 Spearman:  {consistency_new_pct:.4f}" if not np.isnan(consistency_new_pct) else "  份额+Bottom2: N/A (缺少 final_rank_alt_pct)")

    if rank_table is not None:
        try:
            out_path = output_dir / 'final_rank_comparison.csv'
            rank_table.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"\n  ✓ [已导出] 四种算法最终排名 vs 真实排名: {out_path}")
        except Exception:
            pass

    # 总结
    print("\n" + "=" * 80)
    print("综合评估总结")
    print("=" * 80)

    methods_consistency = [
        ('百分比结合法', consistency_pct if not np.isnan(consistency_pct) else -2),
        ('排名结合法', consistency_rank if not np.isnan(consistency_rank) else -2),
        ('排名+Bottom2', consistency_new if not np.isnan(consistency_new) else -2),
        ('份额+Bottom2', consistency_new_pct if not np.isnan(consistency_new_pct) else -2),
    ]
    consistency_winner = max(methods_consistency, key=lambda x: x[1])[0]

    summary_data = {
        '指标': ['稳定性（翻转率）', '抗操纵性（偏移率）', '一致性（最终排名 Spearman）'],
        '百分比结合法': [f'{percentage_flip:.6f}', f'{percentage_robust:.6f}', f'{consistency_pct:.4f}' if not np.isnan(consistency_pct) else 'N/A'],
        '排名结合法': [f'{rank_flip:.6f}', f'{rank_robust:.6f}', f'{consistency_rank:.4f}' if not np.isnan(consistency_rank) else 'N/A'],
        '排名+Bottom2': [f'{new_flip:.6f}', f'{new_robust:.6f}', f'{consistency_new:.4f}' if not np.isnan(consistency_new) else 'N/A'],
        '份额+Bottom2': [f'{new_pct_flip:.6f}', f'{new_pct_robust:.6f}', f'{consistency_new_pct:.4f}' if not np.isnan(consistency_new_pct) else 'N/A'],
        '更优方法': [stability_winner, robust_winner, consistency_winner]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # 导出
    try:
        summary_df.to_csv(output_dir / 'metric_comprehensive_comparison.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ [已导出] metric_comprehensive_comparison.csv")
    except:
        pass
    
    return {
        'stability': {'percentage': percentage_flip, 'rank': rank_flip, 'new': new_flip, 'new_pct': new_pct_flip},
        'robustness': {'percentage': percentage_robust, 'rank': rank_robust, 'new': new_robust, 'new_pct': new_pct_robust},
        'consistency': {
            'survival': consistency_survival,
            'percentage': consistency_pct,
            'rank': consistency_rank,
            'new': consistency_new,
            'new_pct': consistency_new_pct,
        },
        'final_rank_table': rank_table,
    }
