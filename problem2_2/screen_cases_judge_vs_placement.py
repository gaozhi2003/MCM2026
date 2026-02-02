"""
问题二（2）：按「每人每季评委均值排名 vs 最终真实排名」及「评委垫底/靠后周数」筛选案例

题目示例强调：评委多次给最低/垫底分，但观众投票让选手最终名次很好。因此除「评委均值−placement」外，
增加「当周评委垫底周数」n_weeks_judge_last 与「当周评委倒数两名周数」n_weeks_judge_bottom2，
并优先保证题目四例出现在前列。

【当前筛选的潜在问题】
1. “垫底”定义过严：只统计「当周排名 = 当周最差」的周数。Bobby Bones 很多周是倒数第二而非垫底，
   故 n_weeks_judge_last=1，排不到前面。题目“consistently low”更接近“经常在倒数几名”。
2. 误入“名次好+几周垫底”的冠军：如 Emmitt Smith S3 冠军、评委均值 2.6，因早期某几周垫底也被筛进，
   本质是“评委整体好、偶有垫底”，与“观众救上来”不完全一致。
3. 数据/口径差异：题目说 Billy Ray “6 weeks last place”，我们数据里只有 3 周严格垫底，
   可能是数据源或“last place”定义（如含并列/倒数两名）不同。
4. 排序权重任意：score_priority = n_weeks*2 - placement/2 为经验公式，可调。

改进：增加 n_weeks_judge_bottom2（当周排名在倒数两名内），综合排序时同时参考。
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = SCRIPT_DIR

RANKINGS_PATH = DATA_DIR / "new_method_rankings.csv"
WEEKLY_RANKS_PATH = DATA_DIR / "weekly_ranks.csv"
CLEANED_PATH = DATA_DIR / "cleaned_data1.csv"


def load_weekly_judge_ranks() -> pd.DataFrame:
    """加载每周每人评委排名，统一列名为 judge_rank"""
    if RANKINGS_PATH.exists():
        df = pd.read_csv(RANKINGS_PATH)
        if "judge_rank" in df.columns:
            return df[["season", "week", "celebrity_name", "judge_rank"]].copy()
    df = pd.read_csv(WEEKLY_RANKS_PATH)
    if "评委排名" not in df.columns:
        raise ValueError("未找到评委排名列（judge_rank 或 评委排名）")
    df = df.rename(columns={"评委排名": "judge_rank"})
    return df[["season", "week", "celebrity_name", "judge_rank"]].copy()


def load_judge_mean_and_n_weeks_last() -> pd.DataFrame:
    """每人每季：评委均值 + 当周垫底周数 n_weeks_judge_last + 当周倒数两名周数 n_weeks_judge_bottom2"""
    weekly = load_weekly_judge_ranks()
    # 当周该季内评委排名最差、次差（用于 bottom2）
    week_max = weekly.groupby(["season", "week"], as_index=False)["judge_rank"].max()
    week_max = week_max.rename(columns={"judge_rank": "judge_rank_max_that_week"})
    weekly = weekly.merge(week_max, on=["season", "week"], how="left")
    weekly["is_judge_last_that_week"] = weekly["judge_rank"] >= weekly["judge_rank_max_that_week"]

    # 当周是否在倒数两名内：rank >= (max - 1)，即 last 或 second-to-last；max=1 时按 1 算
    weekly["judge_rank_second_worst"] = (weekly["judge_rank_max_that_week"] - 1).clip(lower=1)
    weekly["is_judge_bottom2_that_week"] = weekly["judge_rank"] >= weekly["judge_rank_second_worst"]

    judge_mean = (
        weekly.groupby(["season", "celebrity_name"], as_index=False)["judge_rank"]
        .mean()
        .rename(columns={"judge_rank": "judge_mean_rank"})
    )
    n_last = (
        weekly.groupby(["season", "celebrity_name"], as_index=False)["is_judge_last_that_week"]
        .sum()
        .rename(columns={"is_judge_last_that_week": "n_weeks_judge_last"})
    )
    n_bottom2 = (
        weekly.groupby(["season", "celebrity_name"], as_index=False)["is_judge_bottom2_that_week"]
        .sum()
        .rename(columns={"is_judge_bottom2_that_week": "n_weeks_judge_bottom2"})
    )
    out = judge_mean.merge(n_last, on=["season", "celebrity_name"], how="left")
    out = out.merge(n_bottom2, on=["season", "celebrity_name"], how="left")
    return out


def load_placement_per_season() -> pd.DataFrame:
    """每人每季：最终真实排名 placement（去重，一人一季一行）"""
    df = pd.read_csv(CLEANED_PATH)
    if "placement" not in df.columns:
        raise ValueError("cleaned_data1 缺少 placement 列")
    out = df[["season", "celebrity_name", "placement"]].drop_duplicates(
        subset=["season", "celebrity_name"]
    )
    return out


def main():
    print("=" * 60)
    print("问题二（2）：评委 vs 最终排名 — 案例筛选（含评委垫底周数）")
    print("=" * 60)

    if not CLEANED_PATH.exists():
        print(f"[错误] 缺少 {CLEANED_PATH}")
        return
    if not RANKINGS_PATH.exists() and not WEEKLY_RANKS_PATH.exists():
        print("[错误] 缺少 new_method_rankings.csv 或 weekly_ranks.csv")
        return

    # 1) 每人每季：评委均值 + 评委垫底周数
    print("\n[1] 计算每人每季评委均值排名 + 当周评委垫底周数...")
    judge_stats = load_judge_mean_and_n_weeks_last()
    print(f"    共 {len(judge_stats)} 条 (season, celebrity)")

    # 2) 每人每季最终真实排名
    print("\n[2] 加载每人每季最终真实排名 (placement)...")
    placement = load_placement_per_season()
    print(f"    共 {len(placement)} 条")

    # 3) 合并
    merged = judge_stats.merge(
        placement, on=["season", "celebrity_name"], how="inner"
    )
    merged["diff_judge_placement"] = merged["judge_mean_rank"] - merged["placement"]
    merged["abs_diff"] = merged["diff_judge_placement"].abs()

    # 4) 综合排序：「名次好且评委常垫底/靠后」优先，再 score、abs_diff
    merged["placement_good_and_judge_last_many"] = (
        (merged["placement"] <= 5) & (merged["n_weeks_judge_last"] >= 3)
    )
    merged["score_priority"] = (
        merged["n_weeks_judge_last"].astype(float) * 2
        + merged["n_weeks_judge_bottom2"].astype(float) * 0.5
        - merged["placement"].astype(float) / 2
    )
    merged = merged.sort_values(
        ["placement_good_and_judge_last_many", "score_priority", "abs_diff"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 5) 输出完整表
    out_full = OUT_DIR / "judge_mean_vs_placement_all.csv"
    merged.to_csv(out_full, index=False)
    print(f"\n[输出] 完整表: {out_full}")

    # 6) 优先输出「名次好+评委垫底周数多」的前 N 条（含题目四例）
    top_n = 40
    out_priority = OUT_DIR / "judge_mean_vs_placement_top_discrepancy.csv"
    merged.head(top_n).to_csv(out_priority, index=False)
    print(f"[输出] 综合排序前 {top_n} 条: {out_priority}")

    # 7) 题目四例：单独列出并校验（不参与置顶，仅打印参考）
    PROBLEM_EXAMPLES = [(2, "Jerry Rice"), (4, "Billy Ray Cyrus"), (11, "Bristol Palin"), (27, "Bobby Bones")]
    print("\n--- 题目示例四人 (season, celebrity, 评委均值, placement, 垫底周数, 倒数两名周数, diff) ---")
    for s, c in PROBLEM_EXAMPLES:
        row = merged[(merged["season"] == s) & (merged["celebrity_name"] == c)]
        if len(row) > 0:
            r = row.iloc[0]
            b2 = r.get("n_weeks_judge_bottom2", "-")
            print(f"  {s}, {c}: judge_mean={r['judge_mean_rank']:.2f}, placement={r['placement']}, 垫底周={r['n_weeks_judge_last']}, 倒数两名周={b2}, diff={r['diff_judge_placement']:.2f}")
        else:
            print(f"  [未找到] season={s}, celebrity={c}")

    print("\n--- 综合排序前 15 条 ---")
    disp_cols = ["season", "celebrity_name", "judge_mean_rank", "placement", "n_weeks_judge_last"]
    if "n_weeks_judge_bottom2" in merged.columns:
        disp_cols.append("n_weeks_judge_bottom2")
    disp_cols.append("diff_judge_placement")
    disp = merged[disp_cols].head(15)
    print(disp.to_string(index=False))
    print("\n完成。")


if __name__ == "__main__":
    main()
