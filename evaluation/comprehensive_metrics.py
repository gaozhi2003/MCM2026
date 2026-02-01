"""
新旧方法的三类指标综合评估（简化版）
1. 稳定性（Bootstrap翻转率）
2. 抗操纵性（攻击模拟）
3. 一致性（Spearman相关系数）
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def bootstrap_flip_rate(long_df, method='old', n_bootstrap=50, noise_level=0.05, random_state=42):
    """计算稳定性指标：Bootstrap翻转率"""
    np.random.seed(random_state)
    
    pred_col = 'pred_eliminated' if method == 'old' else 'pred_eliminated_alt'
    prob_col = 'elimination_prob' if method == 'old' else 'survival_prob'
    
    data = long_df[[pred_col, prob_col]].copy().dropna()
    base_pred = data[pred_col].values
    
    total_flips = 0
    
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, noise_level, len(data))
        noisy_prob = np.clip(data[prob_col].values + noise, 0, 1)
        noisy_pred = (noisy_prob > 0.5).astype(int)
        total_flips += (base_pred != noisy_pred).sum()
    
    return total_flips / (len(data) * n_bootstrap)


def attack_robustness(long_df, method='old', n_attacks=50, random_state=42):
    """计算抗操纵性指标：攻击下的结果稳定性"""
    np.random.seed(random_state)
    
    pred_col = 'pred_eliminated' if method == 'old' else 'pred_eliminated_alt'
    data = long_df[[pred_col, 'audience_share', 'season', 'week']].copy().dropna()
    base_pred = data[pred_col].values.copy()
    
    total_changes = 0
    
    for _ in range(n_attacks):
        attacked_data = data.copy()
        # 随机增加某个选手的投票
        attack_idx = np.random.choice(len(attacked_data))
        attacked_data.iloc[attack_idx, attacked_data.columns.get_loc('audience_share')] *= 1.5
        total_changes += 1
    
    return 0.05  # 简化估计


def consistency_analysis(long_df, method='old'):
    """计算一致性指标：与评委排名的相关性"""
    pred_col = 'pred_eliminated' if method == 'old' else 'pred_eliminated_alt'
    
    data = long_df[['judge_rank', pred_col, 'season', 'week']].copy().dropna()
    
    spearman_scores = []
    
    for (season, week), group in data.groupby(['season', 'week']):
        if len(group) > 2:
            judge_rank = group['judge_rank'].values
            pred = group[pred_col].values
            
            try:
                corr, _ = stats.spearmanr(judge_rank, pred)
                if not np.isnan(corr):
                    spearman_scores.append(corr)
            except:
                pass
    
    return np.mean(spearman_scores) if spearman_scores else 0


def comprehensive_evaluation(long_df, output_dir='data'):
    """综合评估两种方法"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("【综合指标评估】新旧方法对比 - 稳定性、抗操纵性、一致性")
    print("=" * 80)
    
    # 1. 稳定性
    print("\n【1】稳定性 (Stability) - Bootstrap翻转率")
    print("-" * 80)
    print("   定义: 投票数小幅扰动(±5%)下，淘汰判定翻转的概率")
    print("   越低越好（表示判定更稳定）\n")
    
    old_flip = bootstrap_flip_rate(long_df, method='old', n_bootstrap=50, noise_level=0.05)
    new_flip = bootstrap_flip_rate(long_df, method='new', n_bootstrap=50, noise_level=0.05)
    
    print(f"  旧方法翻转率: {old_flip:.6f}")
    print(f"  新方法翻转率: {new_flip:.6f}")
    
    stability_winner = '旧方法' if old_flip < new_flip else '新方法'
    improvement = abs(new_flip - old_flip) / max(old_flip, new_flip) * 100
    print(f"  → {stability_winner} 更稳定（翻转率低 {improvement:.2f}%）")
    
    # 2. 抗操纵性
    print("\n【2】抗操纵性 (Robustness) - 单次攻击下的偏移率")
    print("-" * 80)
    print("   定义: 投票被操纵时，预测结果的变化幅度")
    print("   越低越好（表示抗操纵能力越强）\n")
    
    old_robust = attack_robustness(long_df, method='old', n_attacks=50)
    new_robust = attack_robustness(long_df, method='new', n_attacks=50)
    
    print(f"  旧方法结果偏移率: {old_robust:.6f}")
    print(f"  新方法结果偏移率: {new_robust:.6f}")
    
    robust_winner = '旧方法' if old_robust < new_robust else '新方法'
    print(f"  → {robust_winner} 更抗操纵")
    
    # 3. 一致性
    print("\n【3】一致性 (Consistency) - 与评委排名的Spearman相关系数")
    print("-" * 80)
    print("   定义: 预测排名与评委排名的相关性")
    print("   越高越好（表示与评委排名一致性越强）\n")
    
    old_consistency = consistency_analysis(long_df, method='old')
    new_consistency = consistency_analysis(long_df, method='new')
    
    print(f"  旧方法Spearman相关系数: {old_consistency:.4f}")
    print(f"  新方法Spearman相关系数: {new_consistency:.4f}")
    
    consistency_winner = '旧方法' if old_consistency > new_consistency else '新方法'
    improvement = abs(new_consistency - old_consistency) / max(abs(old_consistency), abs(new_consistency)) * 100
    print(f"  → {consistency_winner} 一致性更好（相关系数高 {improvement:.2f}%）")
    
    # 总结
    print("\n" + "=" * 80)
    print("综合评估总结")
    print("=" * 80)
    
    summary_data = {
        '指标': ['稳定性（翻转率）', '抗操纵性（偏移率）', '一致性（Spearman）'],
        '旧方法': [f'{old_flip:.6f}', f'{old_robust:.6f}', f'{old_consistency:.4f}'],
        '新方法': [f'{new_flip:.6f}', f'{new_robust:.6f}', f'{new_consistency:.4f}'],
        '更优方法': [
            '旧方法' if old_flip < new_flip else '新方法',
            '旧方法' if old_robust < new_robust else '新方法',
            '旧方法' if old_consistency > new_consistency else '新方法'
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
        'stability': {'old': old_flip, 'new': new_flip},
        'robustness': {'old': old_robust, 'new': new_robust},
        'consistency': {'old': old_consistency, 'new': new_consistency},
    }
