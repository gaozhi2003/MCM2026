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


def _predict_adaptive_method(long_df):
    """动态权重方法：基于自适应权重的淘汰预测"""
    adaptive_df = apply_adaptive_weight_method(long_df)
    return adaptive_df["pred_eliminated_adaptive"]



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
    elif method == 'adaptive':
        base_pred = _predict_adaptive_method(long_df).values
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
        elif method == 'adaptive':
            noisy_pred = _predict_adaptive_method(noisy_df).values
        
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
    elif method == 'adaptive':
        base_pred = _predict_adaptive_method(long_df).values
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
        elif method == 'adaptive':
            attacked_pred = _predict_adaptive_method(attacked).values
        
        change_rates.append((attacked_pred != base_pred).mean())
    return float(np.mean(change_rates))


def consistency_analysis(long_df, method='percentage'):
    """计算一致性指标：预测排名与实际淘汰排名(placement)的Spearman相关性"""
    if 'placement' not in long_df.columns:
        return 0

    # 根据方法选择排名列
    if method == 'percentage':
        # 百分比结合法：使用 combined_share_rank（值越小=综合得分越高=风险越低）
        if 'combined_share_rank' not in long_df.columns:
            return 0
        rank_col = 'combined_share_rank'
        # placement 越大=淘汰越早=风险越高
        # combined_share_rank 越大=综合得分越低=风险越高
        # 应该正相关
    elif method == 'rank':
        # 排名结合法：使用 combined_rank_final（值越小=综合排名越好=风险越低）
        if 'combined_rank_final' not in long_df.columns:
            return 0
        rank_col = 'combined_rank_final'
        # placement 越大=淘汰越早=风险越高
        # combined_rank_final 越大=综合排名越差=风险越高
        # 应该正相关
    elif method == 'new':
        # 新方法：使用 final_rank_alt（基于末尾两位+评委最低规则的排名）
        if 'final_rank_alt' not in long_df.columns:
            return 0
        rank_col = 'final_rank_alt'
    elif method == 'adaptive':
        # 动态权重方法：使用 adaptive_rank
        if 'adaptive_rank' not in long_df.columns:
            return 0
        rank_col = 'adaptive_rank'
    else:
        return 0

    data = long_df[['placement', rank_col, 'season', 'week']].copy().dropna()
    spearman_scores = []

    for (season, week), group in data.groupby(['season', 'week']):
        if len(group) > 2:
            # placement越大=淘汰越早=风险越高
            # 预测排名越大=风险越高
            actual_rank = group['placement'].values
            pred_rank = group[rank_col].values

            try:
                # 正相关表示：预测排名高的，实际也淘汰得早
                corr, _ = stats.spearmanr(actual_rank, pred_rank)
                if not np.isnan(corr):
                    spearman_scores.append(corr)
            except Exception:
                continue

    return float(np.mean(spearman_scores)) if spearman_scores else 0


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
    adaptive_flip = bootstrap_flip_rate(long_df, method='adaptive', n_bootstrap=50, noise_level=0.05)
    
    print(f"  百分比结合法翻转率: {percentage_flip:.6f}")
    print(f"  排名结合法翻转率: {rank_flip:.6f}")
    print(f"  新方法翻转率: {new_flip:.6f}")
    print(f"  动态权重法翻转率: {adaptive_flip:.6f}")
    
    stability_winner = min(
        [
            ('百分比结合法', percentage_flip),
            ('排名结合法', rank_flip),
            ('新方法', new_flip),
            ('动态权重法', adaptive_flip),
        ],
        key=lambda x: x[1]
    )[0]
    print(f"  → {stability_winner} 更稳定（翻转率最低）")
    
    # 2. 抗操纵性
    print("\n【2】抗操纵性 (Robustness) - 单次攻击下的偏移率")
    print("-" * 80)
    print("   定义: 投票被操纵时，预测结果的变化幅度")
    print("   越低越好（表示抗操纵能力越强）\n")
    
    percentage_robust = attack_robustness(long_df, method='percentage', n_attacks=100)
    rank_robust = attack_robustness(long_df, method='rank', n_attacks=100)
    new_robust = attack_robustness(long_df, method='new', n_attacks=100)
    adaptive_robust = attack_robustness(long_df, method='adaptive', n_attacks=100)
    
    print(f"  百分比结合法结果偏移率: {percentage_robust:.6f}")
    print(f"  排名结合法结果偏移率: {rank_robust:.6f}")
    print(f"  新方法结果偏移率: {new_robust:.6f}")
    print(f"  动态权重法结果偏移率: {adaptive_robust:.6f}")
    
    robust_winner = min(
        [
            ('百分比结合法', percentage_robust),
            ('排名结合法', rank_robust),
            ('新方法', new_robust),
            ('动态权重法', adaptive_robust),
        ],
        key=lambda x: x[1]
    )[0]
    print(f"  → {robust_winner} 更抗操纵（偏移率最低）")
    
    # 3. 一致性
    print("\n【3】一致性 (Consistency) - 与实际淘汰排名的Spearman相关系数")
    print("-" * 80)
    print("   定义: 预测风险分数与实际最终名次(placement)的相关性")
    print("   越高越好（表示预测排名与实际淘汰顺序越一致）\n")
    
    percentage_consistency = consistency_analysis(long_df, method='percentage')
    rank_consistency = consistency_analysis(long_df, method='rank')
    new_consistency = consistency_analysis(long_df, method='new')
    adaptive_consistency = consistency_analysis(long_df, method='adaptive')
    
    print(f"  百分比结合法Spearman相关系数: {percentage_consistency:.4f}")
    print(f"  排名结合法Spearman相关系数: {rank_consistency:.4f}")
    print(f"  新方法Spearman相关系数: {new_consistency:.4f}")
    print(f"  动态权重法Spearman相关系数: {adaptive_consistency:.4f}")
    
    consistency_winner = max(
        [
            ('百分比结合法', percentage_consistency),
            ('排名结合法', rank_consistency),
            ('新方法', new_consistency),
            ('动态权重法', adaptive_consistency),
        ],
        key=lambda x: x[1]
    )[0]
    print(f"  → {consistency_winner} 一致性更好（相关系数最高）")
    
    # 总结
    print("\n" + "=" * 80)
    print("综合评估总结")
    print("=" * 80)
    
    summary_data = {
        '指标': ['稳定性（翻转率）', '抗操纵性（偏移率）', '一致性（Spearman）'],
        '百分比结合法': [f'{percentage_flip:.6f}', f'{percentage_robust:.6f}', f'{percentage_consistency:.4f}'],
        '排名结合法': [f'{rank_flip:.6f}', f'{rank_robust:.6f}', f'{rank_consistency:.4f}'],
        '新方法': [f'{new_flip:.6f}', f'{new_robust:.6f}', f'{new_consistency:.4f}'],
        '动态权重法': [f'{adaptive_flip:.6f}', f'{adaptive_robust:.6f}', f'{adaptive_consistency:.4f}'],
        '更优方法': [
            stability_winner,
            robust_winner,
            consistency_winner
        ]
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
        'stability': {'percentage': percentage_flip, 'rank': rank_flip, 'new': new_flip, 'adaptive': adaptive_flip},
        'robustness': {'percentage': percentage_robust, 'rank': rank_robust, 'new': new_robust, 'adaptive': adaptive_robust},
        'consistency': {'percentage': percentage_consistency, 'rank': rank_consistency, 'new': new_consistency, 'adaptive': adaptive_consistency},
    }
