"""
无监督指标评估（不依赖真实淘汰标签）
用于比较：百分比结合法、排名结合法、新方法、动态权重法
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from evaluation.adaptive_weight_method import apply_adaptive_weight_method


@dataclass
class MethodConfig:
    name: str
    score_col: str
    score_higher_is_better: bool
    pred_col: str | None = None


def _ensure_audience_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "audience_rank" not in df.columns:
        df = df.copy()
        df["audience_rank"] = df.groupby(["season", "week"])["audience_share"].rank(
            ascending=False, method="min"
        )
    return df


def _season_progress(df: pd.DataFrame) -> pd.Series:
    max_week = df.groupby("season")["week"].transform("max")
    denom = (max_week - 1).replace(0, np.nan)
    progress = (df["week"] - 1) / denom
    return progress.fillna(0.5)


def _good_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    return series if higher_is_better else -series


def _get_n_elim(group: pd.DataFrame) -> int:
    if "n_eliminated" in group.columns:
        return int(group["n_eliminated"].iloc[0])
    if "is_eliminated" in group.columns:
        return int(group["is_eliminated"].sum())
    return 0


def _predict_percentage_method(df: pd.DataFrame) -> pd.Series:
    preds = pd.Series(0, index=df.index, dtype=int)
    for (_, _), group in df.groupby(["season", "week"]):
        n_elim = _get_n_elim(group)
        if n_elim <= 0:
            continue
        idx = group.nsmallest(n_elim, "combined_share").index
        preds.loc[idx] = 1
    return preds


def _predict_rank_method(df: pd.DataFrame) -> pd.Series:
    preds = pd.Series(0, index=df.index, dtype=int)
    for (_, _), group in df.groupby(["season", "week"]):
        n_elim = _get_n_elim(group)
        if n_elim <= 0:
            continue
        idx = group.nlargest(n_elim, "combined_rank").index
        preds.loc[idx] = 1
    return preds


def _predict_new_method(df: pd.DataFrame) -> pd.Series:
    preds = pd.Series(0, index=df.index, dtype=int)
    for (_, _), group in df.groupby(["season", "week"]):
        n_elim = _get_n_elim(group)
        if n_elim <= 0:
            continue
        audience_rank = group["audience_rank"]
        judge_rank = group["judge_rank"]
        combined_rank_score = audience_rank + judge_rank
        n_candidates = max(2, n_elim)
        n_candidates = min(n_candidates, len(group))
        candidate_idx = combined_rank_score.nlargest(n_candidates).index
        candidates = group.loc[candidate_idx]
        elim_idx = candidates.nsmallest(n_elim, "judge_total_score").index
        preds.loc[elim_idx] = 1
    return preds


def _predict_adaptive_method(df: pd.DataFrame) -> pd.Series:
    adaptive_df = apply_adaptive_weight_method(df)
    return adaptive_df["pred_eliminated_adaptive"]


def _get_method_config(method: str) -> MethodConfig:
    method = method.lower()
    if method == "percentage":
        return MethodConfig(name="percentage", score_col="combined_share", score_higher_is_better=True)
    if method == "rank":
        return MethodConfig(name="rank", score_col="combined_rank", score_higher_is_better=False)
    if method == "new":
        return MethodConfig(name="new", score_col="combined_rank_score", score_higher_is_better=False)
    if method == "adaptive":
        return MethodConfig(name="adaptive", score_col="combined_score_adaptive", score_higher_is_better=True)
    raise ValueError(f"Unsupported method: {method}")


def _ensure_method_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == "percentage":
        if "combined_share" not in df.columns:
            df["combined_share"] = df["judge_share"] + df["audience_share"]
    elif method == "rank":
        df = _ensure_audience_rank(df)
        if "combined_rank" not in df.columns:
            df["combined_rank"] = df["judge_rank"] + df["audience_rank"]
    elif method == "new":
        df = _ensure_audience_rank(df)
        if "combined_rank_score" not in df.columns:
            df["combined_rank_score"] = df["judge_rank"] + df["audience_rank"]
    elif method == "adaptive":
        if "combined_score_adaptive" not in df.columns or "pred_eliminated_adaptive" not in df.columns:
            df = apply_adaptive_weight_method(df)
    return df


def _get_pred(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "percentage":
        return _predict_percentage_method(df)
    if method == "rank":
        return _predict_rank_method(df)
    if method == "new":
        return _predict_new_method(df)
    if method == "adaptive":
        return _predict_adaptive_method(df)
    raise ValueError(f"Unsupported method: {method}")


def _add_noise_to_audience_share(df: pd.DataFrame, noise_level: float, rng: np.random.Generator) -> pd.DataFrame:
    noisy = df.copy()
    for (season, week), group in noisy.groupby(["season", "week"]):
        mask = (noisy["season"] == season) & (noisy["week"] == week)
        shares = noisy.loc[mask, "audience_share"].values
        noise = rng.normal(0, noise_level, len(shares))
        noisy_shares = np.clip(shares * (1 + noise), 0, None)
        total = noisy_shares.sum()
        if total > 0:
            noisy_shares = noisy_shares / total
        noisy.loc[mask, "audience_share"] = noisy_shares
    return noisy


def _candidate_set(group: pd.DataFrame, method: str) -> Tuple[set, int]:
    n_elim = _get_n_elim(group)
    n_candidates = max(2, n_elim) if n_elim > 0 else 0
    n_candidates = min(n_candidates, len(group))
    if n_candidates == 0:
        return set(), 0

    cfg = _get_method_config(method)
    good_score = _good_score(group[cfg.score_col], cfg.score_higher_is_better)
    candidate_idx = good_score.nsmallest(n_candidates).index
    return set(candidate_idx), n_candidates


def elimination_margin(df: pd.DataFrame, method: str) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)
    cfg = _get_method_config(method)
    preds = _get_pred(df, method)

    margins = []
    for (season, week), group in df.groupby(["season", "week"]):
        n_elim = _get_n_elim(group)
        if n_elim <= 0 or n_elim >= len(group):
            continue
        group = group.copy()
        group["_pred"] = preds.loc[group.index].values
        elim = group[group["_pred"] == 1]
        safe = group[group["_pred"] == 0]
        if elim.empty or safe.empty:
            continue
        good_score = _good_score(group[cfg.score_col], cfg.score_higher_is_better)
        elim_max = good_score.loc[elim.index].max()
        safe_min = good_score.loc[safe.index].min()
        margins.append(safe_min - elim_max)

    if not margins:
        return {"margin_mean": np.nan, "margin_std": np.nan}
    return {
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
    }


def candidate_stability(df: pd.DataFrame, method: str, noise_level: float = 0.05, n_trials: int = 30, random_state: int = 42) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)
    rng = np.random.default_rng(random_state)

    overlap_rates = []
    for _ in range(n_trials):
        noisy = _add_noise_to_audience_share(df, noise_level, rng)
        noisy = _ensure_method_columns(noisy, method)

        for (season, week), group in df.groupby(["season", "week"]):
            base_set, k = _candidate_set(group, method)
            if k == 0:
                continue
            noisy_group = noisy[(noisy["season"] == season) & (noisy["week"] == week)]
            noisy_set, _ = _candidate_set(noisy_group, method)
            if not base_set:
                continue
            overlap = len(base_set & noisy_set) / len(base_set)
            overlap_rates.append(overlap)

    if not overlap_rates:
        return {"candidate_stability": np.nan}
    return {"candidate_stability": float(np.mean(overlap_rates))}


def consensus_rate(df: pd.DataFrame, method: str, tail_pct: float = 0.3) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)
    df = _ensure_audience_rank(df)
    preds = _get_pred(df, method)

    eliminated = df[preds == 1].copy()
    if eliminated.empty:
        return {"consensus_rate": np.nan}

    def _is_bottom(rank: pd.Series) -> pd.Series:
        n = rank.groupby([df.loc[rank.index, "season"], df.loc[rank.index, "week"]]).transform("max")
        threshold = np.ceil((1 - tail_pct) * n)
        return rank >= threshold

    judge_bottom = _is_bottom(eliminated["judge_rank"])
    audience_bottom = _is_bottom(eliminated["audience_rank"])
    both = (judge_bottom & audience_bottom).mean()
    return {"consensus_rate": float(both)}


def weight_utilization(df: pd.DataFrame) -> Dict[str, float]:
    """仅用于动态权重法：与固定50/50法预测差异率"""
    adaptive_df = apply_adaptive_weight_method(df)
    pred_adaptive = adaptive_df["pred_eliminated_adaptive"].values

    fixed = df.copy()
    fixed["combined_score_fixed"] = 0.5 * fixed["audience_share"] + 0.5 * fixed["judge_share"]

    preds = pd.Series(0, index=fixed.index, dtype=int)
    for (_, _), group in fixed.groupby(["season", "week"]):
        n_elim = _get_n_elim(group)
        if n_elim <= 0:
            continue
        idx = group.nsmallest(n_elim, "combined_score_fixed").index
        preds.loc[idx] = 1

    diff_rate = (preds.values != pred_adaptive).mean()
    return {"weight_utilization_diff": float(diff_rate)}


def phase_coherence(df: pd.DataFrame, method: str, early_t: float = 0.33, late_t: float = 0.67) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)
    df = _ensure_audience_rank(df)
    df = df.copy()
    df["progress"] = _season_progress(df)
    preds = _get_pred(df, method)
    df = df[preds == 1].copy()
    if df.empty:
        return {"early_audience_tail_rate": np.nan, "late_judge_tail_rate": np.nan}

    def _bottom_rate(sub: pd.DataFrame, col: str, tail_pct: float = 0.3) -> float:
        if sub.empty:
            return np.nan
        ranks = sub[col]
        n = sub.groupby(["season", "week"])[col].transform("max")
        threshold = np.ceil((1 - tail_pct) * n)
        return float((ranks >= threshold).mean())

    early = df[df["progress"] <= early_t]
    late = df[df["progress"] >= late_t]

    return {
        "early_audience_tail_rate": _bottom_rate(early, "audience_rank"),
        "late_judge_tail_rate": _bottom_rate(late, "judge_rank"),
    }


def trend_protection(df: pd.DataFrame, method: str, window: int = 3) -> Dict[str, float]:
    df = df.copy()
    df = df.sort_values(["season", "celebrity_name", "week"])
    df["audience_share_lag1"] = df.groupby(["season", "celebrity_name"])["audience_share"].shift(1)
    df["audience_share_lag2"] = df.groupby(["season", "celebrity_name"])["audience_share"].shift(2)
    df["trend_up"] = (
        (df["audience_share"] > df["audience_share_lag1"]) &
        (df["audience_share_lag1"] > df["audience_share_lag2"])
    )

    df = _ensure_method_columns(df, method)
    preds = _get_pred(df, method)
    df["pred_elim"] = preds.values

    trend_group = df[df["trend_up"]]
    if trend_group.empty:
        return {"trend_elim_rate": np.nan}

    elim_rate = trend_group["pred_elim"].mean()
    return {"trend_elim_rate": float(elim_rate)}


def edge_protection(df: pd.DataFrame, method: str, top_pct: float = 0.2) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)
    df = _ensure_audience_rank(df)
    preds = _get_pred(df, method)
    df = df.copy()
    df["pred_elim"] = preds.values

    n = df.groupby(["season", "week"])["judge_rank"].transform("max")
    top_thresh = np.ceil(top_pct * n)
    bottom_thresh = np.ceil((1 - top_pct) * n)

    judge_top = df["judge_rank"] <= top_thresh
    judge_bottom = df["judge_rank"] >= bottom_thresh
    audience_top = df["audience_rank"] <= top_thresh
    audience_bottom = df["audience_rank"] >= bottom_thresh

    case_judge_top_audience_bottom = df[judge_top & audience_bottom]
    case_audience_top_judge_bottom = df[audience_top & judge_bottom]

    def _protected_rate(case_df: pd.DataFrame) -> float:
        if case_df.empty:
            return np.nan
        return float(1 - case_df["pred_elim"].mean())

    return {
        "protect_judge_top_audience_bottom": _protected_rate(case_judge_top_audience_bottom),
        "protect_audience_top_judge_bottom": _protected_rate(case_audience_top_judge_bottom),
    }


def elimination_concentration(df: pd.DataFrame, method: str) -> Dict[str, float]:
    preds = _get_pred(df, method)
    df = df.copy()
    df["pred_elim"] = preds.values

    weekly = df.groupby(["season", "week"])["pred_elim"].sum().values
    if len(weekly) == 0:
        return {"elim_std": np.nan, "elim_gini": np.nan}

    std = float(np.std(weekly))
    if np.all(weekly == 0):
        gini = 0.0
    else:
        sorted_x = np.sort(weekly)
        n = len(sorted_x)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(sorted_x) / (n * sorted_x.sum()))

    return {"elim_std": std, "elim_gini": gini}


def compute_unsupervised_metrics(df: pd.DataFrame, method: str) -> Dict[str, float]:
    df = _ensure_method_columns(df, method)

    metrics = {}
    metrics.update(elimination_margin(df, method))
    metrics.update(candidate_stability(df, method))
    metrics.update(consensus_rate(df, method))
    metrics.update(phase_coherence(df, method))
    metrics.update(trend_protection(df, method))
    metrics.update(edge_protection(df, method))
    metrics.update(elimination_concentration(df, method))

    if method == "adaptive":
        metrics.update(weight_utilization(df))
    else:
        metrics["weight_utilization_diff"] = np.nan

    return metrics


def compare_methods(df: pd.DataFrame, methods: List[str] | None = None) -> pd.DataFrame:
    if methods is None:
        methods = ["percentage", "rank", "new", "adaptive"]

    rows = []
    for m in methods:
        metrics = compute_unsupervised_metrics(df, m)
        metrics["method"] = m
        rows.append(metrics)

    result = pd.DataFrame(rows)
    return result
