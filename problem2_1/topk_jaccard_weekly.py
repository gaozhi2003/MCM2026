"""
Top-k 一致性（按周）：用 Jaccard 比较「观众 top-k」与「方法综合 top-k」。
- Top_k(F) = 当周观众排序前 k 人（来自 weekly_ranks 的 观众排名）
- Top_k(S)_百分比法 = 当周百分比法综合排序前 k 人（percentage_method_rankings 的 final_rank）
- Top_k(S)_排名法 = 当周排名法综合排序前 k 人（weekly_ranks 的 总体排名）
Jac_k = |Top_k(F) ∩ Top_k(S)| / |Top_k(F) ∪ Top_k(S)|
判定：哪个方法的平均 Jac_k 更高、更稳定，哪个更偏 fan。
"""

import os
import pandas as pd
import numpy as np

# 路径：脚本在 problem2_1，数据在 data，输出在 problem2_1
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WEEKLY_RANKS_PATH = os.path.join(DATA_DIR, "weekly_ranks.csv")
PERCENTAGE_RANKINGS_PATH = os.path.join(DATA_DIR, "percentage_method_rankings.csv")
OUT_DIR = SCRIPT_DIR

# 使用的 k 列表（常用 1, 3）；只统计当周人数 n >= k 的周
K_LIST = [1, 2, 3, 4, 5]


def top_k_names(df_week: pd.DataFrame, rank_col: str, k: int) -> set:
    """当周按 rank_col 排序，取前 k 人（并列按姓名稳定排序）。返回姓名集合。"""
    n = len(df_week)
    k_use = min(k, n)
    if k_use <= 0:
        return set()
    sorted_df = df_week.sort_values([rank_col, "celebrity_name"]).head(k_use)
    return set(sorted_df["celebrity_name"].tolist())


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard = |A∩B| / |A∪B|。空集与空集返回 0。"""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


def main():
    # 读取数据
    weekly_ranks = pd.read_csv(WEEKLY_RANKS_PATH)
    # 列名：season, week, celebrity_name, 评委排名, 观众排名, 总体排名
    pct_rankings = pd.read_csv(PERCENTAGE_RANKINGS_PATH)
    # 列名：season, week, celebrity_name, judge_rank, audience_share_rank, final_rank, ...

    # 合并两表：当周同一批选手，同时有 观众排名、总体排名、final_rank
    merged = weekly_ranks.merge(
        pct_rankings[["season", "week", "celebrity_name", "final_rank"]],
        on=["season", "week", "celebrity_name"],
        how="inner",
    )

    rows = []
    for (season, week), g in merged.groupby(["season", "week"]):
        n = len(g)
        for k in K_LIST:
            if n < k:
                continue  # 只统计 n >= k 的周
            top_fan = top_k_names(g, "观众排名", k)
            top_ranking = top_k_names(g, "总体排名", k)
            top_pct = top_k_names(g, "final_rank", k)
            jac_pct = jaccard(top_fan, top_pct)
            jac_rank = jaccard(top_fan, top_ranking)
            rows.append({
                "season": season,
                "week": week,
                "k": k,
                "n_contestants": n,
                "jac_fan_vs_percentage": jac_pct,
                "jac_fan_vs_ranking": jac_rank,
            })

    weekly_df = pd.DataFrame(rows)

    weekly_path = os.path.join(OUT_DIR, "topk_jaccard_weekly.csv")
    weekly_df.to_csv(weekly_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 周级 Jaccard: {weekly_path}")

    # 汇总：按 k 分组（周表已只含 n>=k 的周）
    summary_rows = []
    for k in K_LIST:
        sub = weekly_df[weekly_df["k"] == k]
        if sub.empty:
            continue
        mean_pct = sub["jac_fan_vs_percentage"].mean()
        std_pct = sub["jac_fan_vs_percentage"].std()
        mean_rank = sub["jac_fan_vs_ranking"].mean()
        std_rank = sub["jac_fan_vs_ranking"].std()
        summary_rows.append({
            "k": k,
            "n_weeks": len(sub),
            "mean_jac_fan_vs_percentage": mean_pct,
            "std_jac_fan_vs_percentage": std_pct if not np.isnan(std_pct) else 0,
            "mean_jac_fan_vs_ranking": mean_rank,
            "std_jac_fan_vs_ranking": std_rank if not np.isnan(std_rank) else 0,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "topk_jaccard_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 汇总: {summary_path}")

    # 判定与简要报告
    report_path = os.path.join(OUT_DIR, "topk_jaccard_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Top-k 一致性（Jaccard）判定报告\n")
        f.write("=" * 50 + "\n")
        for _, r in summary_df.iterrows():
            k = int(r["k"])
            f.write(f"\nk = {k}（仅统计当周人数>={k}的周，共 {int(r['n_weeks'])} 周）\n")
            f.write(f"  百分比法: 平均 Jac = {r['mean_jac_fan_vs_percentage']:.4f}, 标准差 = {r['std_jac_fan_vs_percentage']:.4f}\n")
            f.write(f"  排名法:   平均 Jac = {r['mean_jac_fan_vs_ranking']:.4f}, 标准差 = {r['std_jac_fan_vs_ranking']:.4f}\n")
            if r["mean_jac_fan_vs_percentage"] > r["mean_jac_fan_vs_ranking"]:
                f.write(f"  => 该 k 下百分比法平均 Jac 更高，更偏 fan。\n")
            elif r["mean_jac_fan_vs_ranking"] > r["mean_jac_fan_vs_percentage"]:
                f.write(f"  => 该 k 下排名法平均 Jac 更高，更偏 fan。\n")
            else:
                f.write(f"  => 两者平均 Jac 相同。\n")
        f.write("\n说明: Jac_k = 观众 top-k 与方法 top-k 的 Jaccard；平均越高、越稳定，该方法越偏 fan。\n")
    print(f"[已导出] 报告: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
