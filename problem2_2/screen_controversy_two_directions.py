"""
问题二（2）：「低评委走得远」争议案例筛选（不依赖真实 fan votes）

只筛一类：评委常给倒数、但该周安全，且最终名次靠前（观众把评委不喜欢的人一路投活，如 Bristol Palin、Bobby Bones）。

只用「评委排名 + 是否被淘汰/安全」：
  - low_judge_safe_weeks = count(评委倒数 k 且 该周安全)
  阈值：low_judge_safe_pct ≥ 40%，且 placement≤5
"""

import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = SCRIPT_DIR

RANKINGS_PATH = DATA_DIR / "new_method_rankings.csv"
WEEKLY_RANKS_PATH = DATA_DIR / "weekly_ranks.csv"
CLEANED_PATH = DATA_DIR / "cleaned_data1.csv"

JUDGE_BOTTOM_K = 2  # 评委倒数 k 名（k=1 垫底，k=2 倒数两名）
LOW_JUDGE_SAFE_PCT_MIN = 0.40  # 只按比例筛：≥40%
PLACEMENT_GOOD_MAX = 5  # 「走得远」：placement ≤ 5


def load_weekly_judge() -> pd.DataFrame:
    """每周每人：judge_rank，当周人数 n，是否倒数 k（judge_bottom_k）"""
    if RANKINGS_PATH.exists():
        df = pd.read_csv(RANKINGS_PATH)[["season", "week", "celebrity_name", "judge_rank"]]
    else:
        df = pd.read_csv(WEEKLY_RANKS_PATH)
        df = df.rename(columns={"评委排名": "judge_rank"})[["season", "week", "celebrity_name", "judge_rank"]]
    week_n = df.groupby(["season", "week"], as_index=False).size().rename(columns={"size": "n"})
    df = df.merge(week_n, on=["season", "week"], how="left")
    week_max = df.groupby(["season", "week"], as_index=False)["judge_rank"].max().rename(
        columns={"judge_rank": "judge_rank_max"}
    )
    df = df.merge(week_max, on=["season", "week"], how="left")
    # 倒数 k 名：rank >= max - (k-1)，且不小于 1
    df["judge_rank_k_th_worst"] = (df["judge_rank_max"] - (JUDGE_BOTTOM_K - 1)).clip(lower=1)
    df["judge_bottom_k"] = (df["judge_rank"] >= df["judge_rank_k_th_worst"]).astype(int)
    return df


def load_exit_week_and_placement() -> pd.DataFrame:
    """每人每季：exit_week（0 表示未淘汰/冠军），placement"""
    df = pd.read_csv(CLEANED_PATH)
    out = df[["season", "celebrity_name", "placement", "exit_week"]].drop_duplicates(
        subset=["season", "celebrity_name"]
    )
    out["exit_week"] = pd.to_numeric(out["exit_week"], errors="coerce").fillna(0)
    return out


def layer1_weekly_with_safe(weekly_judge: pd.DataFrame, exit_placement: pd.DataFrame) -> pd.DataFrame:
    """在每周数据上加上：该周是否安全 safe，该周是否被淘汰 eliminated_this_week"""
    weekly = weekly_judge.merge(
        exit_placement[["season", "celebrity_name", "exit_week"]],
        on=["season", "celebrity_name"],
        how="left",
    )
    weekly["exit_week"] = weekly["exit_week"].fillna(0)
    weekly["eliminated_this_week"] = (weekly["exit_week"] > 0) & (weekly["week"] == weekly["exit_week"])
    weekly["safe"] = (~weekly["eliminated_this_week"]).astype(int)
    return weekly


def layer1_aggregate(weekly: pd.DataFrame, exit_placement: pd.DataFrame) -> pd.DataFrame:
    """每人每季：low_judge_safe_weeks, n_weeks_participated, placement"""
    low_safe = (
        weekly.loc[weekly["judge_bottom_k"] == 1]
        .groupby(["season", "celebrity_name"], as_index=False)["safe"]
        .sum()
        .rename(columns={"safe": "low_judge_safe_weeks"})
    )
    n_weeks = (
        weekly.groupby(["season", "celebrity_name"], as_index=False)["week"]
        .count()
        .rename(columns={"week": "n_weeks_participated"})
    )
    out = n_weeks.merge(
        exit_placement[["season", "celebrity_name", "placement"]],
        on=["season", "celebrity_name"],
        how="left",
    )
    out = out.merge(low_safe, on=["season", "celebrity_name"], how="left")
    out["low_judge_safe_weeks"] = out["low_judge_safe_weeks"].fillna(0).astype(int)
    out["low_judge_safe_pct"] = out["low_judge_safe_weeks"] / out["n_weeks_participated"].replace(0, np.nan)
    return out


def filter_low_judge_went_far(l1: pd.DataFrame) -> pd.DataFrame:
    """低评委走得远：low_judge_safe_pct ≥ 40%，且 placement≤5"""
    cond_pct = l1["low_judge_safe_pct"] >= LOW_JUDGE_SAFE_PCT_MIN
    cond_far = l1["placement"] <= PLACEMENT_GOOD_MAX
    mask = cond_pct & cond_far
    out = l1.loc[mask].copy()
    out = out.sort_values(
        ["low_judge_safe_pct", "low_judge_safe_weeks", "placement"],
        ascending=[False, False, True],
    )
    return out


def main():
    print("=" * 60)
    print("问题二（2）：低评委走得远 — 争议案例筛选")
    print("=" * 60)

    if not CLEANED_PATH.exists():
        print(f"[错误] 缺少 {CLEANED_PATH}")
        return
    if not RANKINGS_PATH.exists() and not WEEKLY_RANKS_PATH.exists():
        print("[错误] 缺少 new_method_rankings.csv 或 weekly_ranks.csv")
        return

    exit_placement = load_exit_week_and_placement()
    weekly_judge = load_weekly_judge()
    weekly = layer1_weekly_with_safe(weekly_judge, exit_placement)
    l1 = layer1_aggregate(weekly, exit_placement)

    result = filter_low_judge_went_far(l1)
    result_top10 = result.head(10)
    out_path = OUT_DIR / "controversy_low_judge_went_far.csv"
    result_top10.to_csv(out_path, index=False)
    print(f"\n低评委走得远（前 10 条）: {len(result_top10)} 条 -> {out_path.name}")

    examples = [(2, "Jerry Rice"), (4, "Billy Ray Cyrus"), (11, "Bristol Palin"), (27, "Bobby Bones")]
    print("\n--- 题目四例是否在筛出结果中 ---")
    for s, c in examples:
        row = result[(result["season"] == s) & (result["celebrity_name"] == c)]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {s}, {c}: placement={r['placement']}, low_judge_safe_weeks={r['low_judge_safe_weeks']}, low_judge_safe_pct={r['low_judge_safe_pct']:.2f}")
        else:
            print(f"  [未筛出] {s}, {c}")

    l1["low_judge_went_far"] = (l1["low_judge_safe_pct"] >= LOW_JUDGE_SAFE_PCT_MIN) & (l1["placement"] <= PLACEMENT_GOOD_MAX)
    out_full = OUT_DIR / "controversy_layer1_full.csv"
    l1.to_csv(out_full, index=False)
    print(f"\n完整表（含 low_judge_went_far 标记）: {out_full.name}")

    print("\n完成。")


if __name__ == "__main__":
    main()
