"""不确定性评估模块"""

from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_prediction_intervals(predictions, errors, confidence=0.95):
    """计算预测区间"""
    from scipy import stats
    
    std_error = np.std(errors)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    margin_of_error = z_score * std_error
    
    intervals = {
        'predictions': predictions,
        'lower_bound': predictions - margin_of_error,
        'upper_bound': predictions + margin_of_error,
        'margin_of_error': margin_of_error,
        'confidence_level': confidence
    }
    
    return intervals


def calculate_uncertainty_metrics(residuals):
    """计算不确定性指标"""
    metrics = {
        'mse': np.mean(residuals ** 2),
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'mae': np.mean(np.abs(residuals)),
        'std': np.std(residuals),
        'variance': np.var(residuals)
    }
    
    return metrics


def quantify_parameter_uncertainty(model, X_test):
    """量化参数不确定性"""
    # 这里可以实现 Bootstrap 或 Bayesian 方法
    # 示例实现
    
    return {
        'method': 'placeholder',
        'uncertainty': None
    }


def entropy_uncertainty(vote_share: pd.Series) -> float:
    """用熵衡量不确定性（越大越不确定）"""
    share = vote_share.clip(lower=1e-12)
    return float(-(share * np.log(share)).sum())


def weekly_uncertainty(long_df: pd.DataFrame, vote_share_col: str = "vote_share") -> pd.DataFrame:
    """按赛季-周计算不确定性"""
    grouped = long_df.groupby(["season", "week"])[vote_share_col].apply(entropy_uncertainty).reset_index(name="entropy")
    return grouped


def analyze_vote_share_intervals(
    long_df: pd.DataFrame,
    feature_cols: list[str],
    model_class,
    train_mask: pd.Series | None = None,
    vote_share_col: str = "audience_share",
    confidence: float = 0.95,
    n_bootstrap: int = 500,
    random_state: int = 42
) -> dict:
    """使用模型Bootstrap分析每个选手每周的投票比例不确定性
    
    方法：
    1. 对训练样本有放回重采样
    2. 重新训练观众投票预测模型
    3. 对全体样本预测投票比例
    4. 统计每个选手-周的Bootstrap分布区间与方差
    
    Parameters
    ----------
    long_df : pd.DataFrame
        包含特征与标签的长表
    feature_cols : list[str]
        模型特征列
    model_class : type
        可实例化的模型类（例如 VoteShareModel）
    train_mask : pd.Series | None
        训练样本掩码（与 long_df 对齐），若为 None 则使用全量样本
    vote_share_col : str
        原始投票比例列名，用于输出对比
    confidence : float
        置信度，默认0.95
    n_bootstrap : int
        Bootstrap重采样次数，默认500
    random_state : int
        随机种子
    
    Returns
    -------
    dict
        包含每个选手-周的区间和不确定性指标
    """
    if train_mask is None:
        train_mask = pd.Series(True, index=long_df.index)

    train_df = long_df.loc[train_mask]
    if train_df.empty:
        return {
            'celeb_week_level': pd.DataFrame(),
            'mean_interval_width': np.nan,
            'mean_variance': np.nan,
            'summary': {}
        }

    n_rows = len(long_df)
    rng = np.random.default_rng(random_state)

    bootstrap_preds = np.zeros((n_bootstrap, n_rows), dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.choice(train_df.index.to_numpy(), size=len(train_df), replace=True)
        boot_df = long_df.loc[sample_idx]

        model = model_class()
        model.fit(boot_df[feature_cols], boot_df["is_eliminated"])

        elimination_prob = model.predict(long_df[feature_cols])
        survival_prob = pd.Series(1 - elimination_prob, index=long_df.index)
        audience_share = survival_prob.groupby([long_df["season"], long_df["week"]]).transform(
            lambda s: s / (s.sum() + 1e-10)
        )
        bootstrap_preds[b, :] = audience_share.to_numpy()

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_preds, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_preds, upper_percentile, axis=0)
    interval_width = ci_upper - ci_lower

    mean_bootstrap = bootstrap_preds.mean(axis=0)
    std_bootstrap = bootstrap_preds.std(axis=0)

    celeb_week_df = long_df[["season", "week", "celebrity_name", vote_share_col]].copy()
    celeb_week_df = celeb_week_df.rename(columns={vote_share_col: "vote_share"})
    celeb_week_df["bootstrap_mean"] = mean_bootstrap
    celeb_week_df["bootstrap_std"] = std_bootstrap
    celeb_week_df["bootstrap_variance"] = std_bootstrap ** 2
    celeb_week_df["interval_lower"] = ci_lower
    celeb_week_df["interval_upper"] = ci_upper
    celeb_week_df["interval_width"] = interval_width
    celeb_week_df["n_bootstrap"] = n_bootstrap
    celeb_week_df["confidence_level"] = confidence

    summary = {
        'mean_interval_width': float(np.mean(interval_width)),
        'std_interval_width': float(np.std(interval_width)),
        'mean_variance': float(np.mean(std_bootstrap ** 2)),
        'mean_std': float(np.mean(std_bootstrap)),
        'total_observations': int(n_rows),
        'total_weeks': int(long_df.groupby(["season", "week"]).ngroups)
    }

    return {
        'celeb_week_level': celeb_week_df,
        'mean_interval_width': float(np.mean(interval_width)),
        'mean_variance': float(np.mean(std_bootstrap ** 2)),
        'summary': summary
    }


def _analyze_uncertainty_reasons(long_df: pd.DataFrame, high_uncertainty_weeks: pd.DataFrame) -> dict:
    """分析高不确定性周的原因"""
    reasons = {}
    
    for _, row in high_uncertainty_weeks.iterrows():
        season, week = int(row['season']), int(row['week'])
        week_data = long_df[(long_df['season'] == season) & (long_df['week'] == week)]
        
        # 可能的原因：
        # 1. 参赛人数少
        # 2. 评分分布均匀（竞争激烈）
        # 3. 并列现象多
        
        n_contestants = len(week_data)
        score_std = week_data['judge_total_score'].std()
        n_ties = week_data['judge_total_score'].duplicated().sum()
        
        reasons[f'Season {season}, Week {week}'] = {
            'n_contestants': int(n_contestants),
            'score_std': float(score_std),
            'n_ties': int(n_ties),
            'reason': _determine_reason(n_contestants, score_std, n_ties)
        }
    
    return reasons


def _determine_reason(n_contestants: int, score_std: float, n_ties: int) -> str:
    """判断不确定性的主要原因"""
    if n_contestants <= 2:
        return "参赛人数过少"
    elif score_std < 2.0:
        return "评分分布均匀（竞争激烈）"
    elif n_ties > n_contestants * 0.3:
        return "并列现象较多"
    else:
        return "综合因素"

