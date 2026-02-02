"""
问题二：针对 10 个争议案例，用与 three_method_comparison 相同的逻辑计算四种方法结果并对比

四种方法：
  (i)   全份额（百分比法）：按 combined_share 排名，直接淘汰末位
  (ii)  全排名（排名法）：按 judge_rank + audience_rank 综合排名，直接淘汰末位
  (iii) 排名 + Bottom 2 + 评委裁决：综合排名倒数两名候选，评委分数最低者淘汰
  (iv)  份额 + Bottom 2 + 评委裁决：综合份额最低两名候选，评委分数最低者淘汰

本脚本：构建 long_df → 调用 _compute_all_methods_final_ranks 计算四种方法的预测最终名次
       → 筛选 10 个争议案例 → 输出对比表

依赖：main.py 已生成 percentage_method_rankings, ranking_method_rankings, new_method_rankings,
     new_method_rankings_pct, weekly_shares, cleaned_data1
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))

CONTROVERSY_PATH = SCRIPT_DIR / "controversy_low_judge_went_far.csv"
WEEKLY_SHARES_PATH = DATA_DIR / "weekly_shares.csv"
PERCENTAGE_RANKINGS_PATH = DATA_DIR / "percentage_method_rankings.csv"
RANKING_RANKINGS_PATH = DATA_DIR / "ranking_method_rankings.csv"
NEW_METHOD_RANKINGS_PATH = DATA_DIR / "new_method_rankings.csv"
NEW_METHOD_RANKINGS_PCT_PATH = DATA_DIR / "new_method_rankings_pct.csv"
CLEANED_PATH = DATA_DIR / "cleaned_data1.csv"


def build_long_df():
    """从数据文件合并构建 long_df，包含四种方法的周级排名"""
    import pandas as pd

    base = pd.read_csv(NEW_METHOD_RANKINGS_PATH)
    shares = pd.read_csv(WEEKLY_SHARES_PATH)
    df = base.merge(
        shares[["season", "week", "celebrity_name", "judge_share", "audience_share", "combined_share"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    # 百分比 + Bottom 2 的周级排名
    if NEW_METHOD_RANKINGS_PCT_PATH.exists():
        new_pct = pd.read_csv(NEW_METHOD_RANKINGS_PCT_PATH)
        df = df.merge(
            new_pct[["season", "week", "celebrity_name", "final_rank_alt_pct"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
    pct = pd.read_csv(PERCENTAGE_RANKINGS_PATH)
    pct = pct.rename(columns={"final_rank": "combined_share_rank"})
    df = df.merge(
        pct[["season", "week", "celebrity_name", "combined_share_rank"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    rank_df = pd.read_csv(RANKING_RANKINGS_PATH)
    rank_df = rank_df.rename(columns={"final_rank": "combined_rank_final"})
    df = df.merge(
        rank_df[["season", "week", "celebrity_name", "combined_rank_final"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    cleaned = pd.read_csv(CLEANED_PATH)
    needed = ["season", "week", "celebrity_name", "placement", "exit_week", "is_eliminated", "n_eliminated"]
    if all(c in cleaned.columns for c in needed):
        df = df.merge(
            cleaned[needed].drop_duplicates(subset=["season", "week", "celebrity_name"]),
            on=["season", "week", "celebrity_name"],
            how="left",
        )
    else:
        df["is_eliminated"] = df.get("pred_eliminated_alt", 0)
        df["placement"] = None
        df["n_eliminated"] = df.groupby(["season", "week"])["is_eliminated"].transform("sum")
    if "combined_rank" not in df.columns:
        df["combined_rank"] = df["judge_rank"] + df["audience_rank"]
    return df


def main():
    import pandas as pd
    from evaluation.comprehensive_metrics import _compute_all_methods_final_ranks

    print("=" * 70)
    print("问题二：10 个争议案例 — 四种方法结果对比（按评估逻辑计算）")
    print("=" * 70)

    if not CONTROVERSY_PATH.exists():
        print(f"[错误] 缺少 {CONTROVERSY_PATH}，请先运行 screen_controversy_two_directions.py")
        return
    for p, name in [
        (WEEKLY_SHARES_PATH, "weekly_shares.csv"),
        (PERCENTAGE_RANKINGS_PATH, "percentage_method_rankings.csv"),
        (RANKING_RANKINGS_PATH, "ranking_method_rankings.csv"),
        (NEW_METHOD_RANKINGS_PATH, "new_method_rankings.csv"),
        (NEW_METHOD_RANKINGS_PCT_PATH, "new_method_rankings_pct.csv"),
        (CLEANED_PATH, "cleaned_data1.csv"),
    ]:
        if not p.exists():
            print(f"[错误] 缺少 {name}，请先运行 main.py")
            return

    print("\n[1] 构建 long_df（合并四种方法数据）...")
    long_df = build_long_df()
    if "is_eliminated" not in long_df.columns or long_df["is_eliminated"].isna().all():
        long_df["is_eliminated"] = long_df.get("pred_eliminated_alt", 0)
    print(f"    记录数: {len(long_df)}")

    print("\n[2] 计算四种方法的预测最终名次...")
    rank_table = _compute_all_methods_final_ranks(long_df)
    if rank_table is None:
        print("[错误] 无法计算四种方法最终排名")
        return

    rank_cols = ["pred_final_rank_pct", "pred_final_rank_rank", "pred_final_rank_new", "pred_final_rank_new_pct"]
    rank_cols = [c for c in rank_cols if c in rank_table.columns]

    controversy = pd.read_csv(CONTROVERSY_PATH)
    merge_cols = ["season", "celebrity_name"]
    controversy_sub = controversy.drop(columns=["placement"], errors="ignore")
    df = controversy_sub.merge(
        rank_table[merge_cols + ["placement"] + rank_cols],
        on=merge_cols,
        how="inner",
    )

    rename_map = {
        "placement": "实际名次",
        "pred_final_rank_pct": "全份额法",
        "pred_final_rank_rank": "全排名法",
        "pred_final_rank_new": "排名+Bottom2",
        "pred_final_rank_new_pct": "份额+Bottom2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 四法一致：四种方法结果完全相同
    method_cols = [rename_map[c] for c in rank_cols if c in rename_map]
    if len(method_cols) >= 2:
        df["四法一致"] = df[method_cols].apply(lambda row: row.nunique() == 1, axis=1)
    else:
        df["四法一致"] = True
    df["与实际一致"] = df.apply(
        lambda r: all(r.get(c) == r["实际名次"] for c in method_cols if c in r),
        axis=1
    )

    out_path = OUT_DIR / "controversy_three_methods_comparison.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[输出] {out_path.name}")

    disp_cols = ["season", "celebrity_name", "实际名次"] + method_cols + ["四法一致", "与实际一致"]
    disp_cols = [c for c in disp_cols if c in df.columns]
    print("\n--- 10 个争议案例在四种方法下的最终名次 ---")
    print(df[disp_cols].to_string(index=False))

    n_same_all = df["四法一致"].sum()
    n_match_actual = df["与实际一致"].sum()
    print(f"\n四法结果完全一致: {n_same_all}/10")
    print(f"四法均与实际名次一致: {n_match_actual}/10")

    diff_cases = df[~df["四法一致"]]
    if len(diff_cases) > 0:
        print("\n--- 四种方法结果存在差异的案例 ---")
        for _, r in diff_cases.iterrows():
            parts = [f"实际={r['实际名次']}"]
            for c in method_cols:
                if c in r:
                    parts.append(f"{c}={r[c]}")
            print(f"  S{r['season']} {r['celebrity_name']}: " + ", ".join(parts))

    print("\n完成。")


if __name__ == "__main__":
    main()
