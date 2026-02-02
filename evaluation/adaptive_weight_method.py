"""
动态权重淘汰方法（基础版）

核心思想：
- 观众权重随赛季进度从高到低（S型曲线）
- 评委权重 = 1 - 观众权重
- 综合分数 = w_audience * audience_share + w_judge * judge_share
- 每周选择综合分数最低的候选人，再按评委评分淘汰
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AdaptiveWeightParams:
    base_audience: float = 0.4
    amplitude: float = 0.2
    steepness: float = 5.0
    midpoint: float = 0.5
    min_candidates: int = 2


def _season_progress(long_df: pd.DataFrame) -> pd.Series:
    """计算每条记录的赛季进度 t ∈ [0,1]。"""
    max_week = long_df.groupby("season")["week"].transform("max")
    denom = (max_week - 1).replace(0, np.nan)
    progress = (long_df["week"] - 1) / denom
    return progress.fillna(0.5)


def _audience_weight(progress: pd.Series, params: AdaptiveWeightParams) -> pd.Series:
    """S型曲线观众权重。"""
    return params.base_audience + params.amplitude / (
        1 + np.exp(params.steepness * (progress - params.midpoint))
    )


def apply_adaptive_weight_method(
    long_df: pd.DataFrame,
    params: AdaptiveWeightParams | None = None,
    audience_share_col: str = "audience_share",
    judge_share_col: str = "judge_share",
    judge_score_col: str = "judge_total_score",
    n_elim_col: str = "n_eliminated",
) -> pd.DataFrame:
    """
    在 long_df 上计算动态权重综合分数，并输出预测淘汰与排名。

    必需列：season, week, audience_share, judge_share, judge_total_score, n_eliminated
    输出列：
      - progress
      - weight_audience
      - weight_judge
      - combined_score_adaptive
      - pred_eliminated_adaptive
      - adaptive_rank
    """
    if params is None:
        params = AdaptiveWeightParams()

    required = {"season", "week", audience_share_col, judge_share_col, judge_score_col, n_elim_col}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = long_df.copy()
    df["progress"] = _season_progress(df)
    df["weight_audience"] = _audience_weight(df["progress"], params)
    df["weight_judge"] = 1 - df["weight_audience"]
    df["combined_score_adaptive"] = (
        df["weight_audience"] * df[audience_share_col]
        + df["weight_judge"] * df[judge_share_col]
    )

    df["pred_eliminated_adaptive"] = 0

    def _predict_week(group: pd.DataFrame) -> pd.DataFrame:
        n_elim = int(group[n_elim_col].iloc[0])
        if n_elim <= 0:
            return group

        n_candidates = max(params.min_candidates, n_elim)
        n_candidates = min(n_candidates, len(group))

        candidate_idx = group.nsmallest(n_candidates, "combined_score_adaptive").index
        candidates = group.loc[candidate_idx]
        elim_idx = candidates.nsmallest(n_elim, judge_score_col).index
        group.loc[elim_idx, "pred_eliminated_adaptive"] = 1
        return group

    df = df.groupby(["season", "week"], group_keys=False).apply(_predict_week)

    # 排名（1为最好），淘汰者统一为最差名次
    df["adaptive_rank"] = df.groupby(["season", "week"])["combined_score_adaptive"].rank(
        ascending=False, method="min"
    )

    def _assign_rank(group: pd.DataFrame) -> pd.DataFrame:
        worst_rank = len(group)
        elim_mask = group["pred_eliminated_adaptive"] == 1
        group.loc[elim_mask, "adaptive_rank"] = worst_rank
        return group

    df = df.groupby(["season", "week"], group_keys=False).apply(_assign_rank)
    return df
