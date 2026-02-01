"""
分歧情形下的「跟谁走」：只统计 judge 与 fan 分歧大的周，看淘汰的人里 fan 更喜欢 vs judge 更喜欢，
判定哪种方法更常「救下 fan 支持者」→ 更偏 fan。
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = SCRIPT_DIR

WEEKLY_RANKS_PATH = os.path.join(DATA_DIR, "weekly_ranks.csv")
PERCENTAGE_RANKINGS_PATH = os.path.join(DATA_DIR, "percentage_method_rankings.csv")
RANKING_RANKINGS_PATH = os.path.join(DATA_DIR, "ranking_method_rankings.csv")

# 分歧周：取 |rankJ - rankF| 均值最大的前 TOP_PCT 的周（0.20=前20% 样本多，0.10=前10% 更聚焦高分歧）
# 多百分比展示：对以下每个比例分别计算并在控制台输出
TOP_PCT_LIST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# 文件导出仅针对该默认比例（分歧周列表、淘汰明细、汇总 CSV、报告 TXT）
DEFAULT_TOP_PCT = 0.10


def run_for_top_pct(top_pct, week_div, wr_merged, n_weeks):
    """对给定 top_pct 计算分歧周集合及淘汰统计。"""
    n_top = max(1, int(np.ceil(n_weeks * top_pct)))
    divergent_weeks = set(
        tuple(r) for _, r in week_div.head(n_top)[["season", "week"]].iterrows()
    )
    wr = wr_merged.copy()
    wr["is_divergent"] = wr.apply(
        lambda r: (r["season"], r["week"]) in divergent_weeks, axis=1
    )
    div_only = wr[wr["is_divergent"]].copy()
    div_only["fan_preferred"] = div_only["观众排名"] < div_only["评委排名"]
    div_only["judge_preferred"] = div_only["评委排名"] < div_only["观众排名"]

    elim_pct = div_only[div_only["elim_pct"] == 1]
    elim_rank = div_only[div_only["elim_rank"] == 1]

    n_elim_pct = len(elim_pct)
    n_fan_pct = elim_pct["fan_preferred"].sum()
    n_judge_pct = elim_pct["judge_preferred"].sum()
    n_elim_rank_n = len(elim_rank)
    n_fan_rank = elim_rank["fan_preferred"].sum()
    n_judge_rank = elim_rank["judge_preferred"].sum()

    r_fan_pct = n_fan_pct / n_elim_pct if n_elim_pct else 0
    r_fan_rank = n_fan_rank / n_elim_rank_n if n_elim_rank_n else 0

    return {
        "n_top": n_top,
        "divergent_weeks": divergent_weeks,
        "div_only": div_only,
        "week_div": week_div,
        "n_elim_pct": n_elim_pct,
        "n_fan_pct": n_fan_pct,
        "n_judge_pct": n_judge_pct,
        "r_fan_pct": r_fan_pct,
        "n_elim_rank": n_elim_rank_n,
        "n_fan_rank": n_fan_rank,
        "n_judge_rank": n_judge_rank,
        "r_fan_rank": r_fan_rank,
    }


def main():
    # 1. 周级评委/观众排名
    weekly_ranks = pd.read_csv(WEEKLY_RANKS_PATH)
    weekly_ranks["rank_diff"] = (weekly_ranks["评委排名"] - weekly_ranks["观众排名"]).abs()

    # 2. 每周分歧度
    week_div = (
        weekly_ranks.groupby(["season", "week"])["rank_diff"]
        .mean()
        .reset_index()
        .rename(columns={"rank_diff": "divergence"})
    )
    week_div = week_div.sort_values("divergence", ascending=False).reset_index(drop=True)
    n_weeks = len(week_div)

    # 3. 两种方法的淘汰标记
    pct_df = pd.read_csv(PERCENTAGE_RANKINGS_PATH)[
        ["season", "week", "celebrity_name", "pred_eliminated"]
    ]
    pct_df = pct_df.rename(columns={"pred_eliminated": "elim_pct"})
    rank_df = pd.read_csv(RANKING_RANKINGS_PATH)[
        ["season", "week", "celebrity_name", "pred_eliminated"]
    ]
    rank_df = rank_df.rename(columns={"pred_eliminated": "elim_rank"})

    wr_base = weekly_ranks[["season", "week", "celebrity_name", "评委排名", "观众排名"]].copy()
    wr_merged = wr_base.merge(pct_df, on=["season", "week", "celebrity_name"], how="left")
    wr_merged = wr_merged.merge(rank_df, on=["season", "week", "celebrity_name"], how="left")

    # 4. 多百分比：控制台输出
    print("分歧情形下「跟谁走」—— 多百分比展示（控制台）")
    print("=" * 60)
    print("说明: 分歧周 = 当周 |评委排名-观众排名| 均值 最大的前 X% 周；")
    print("      淘汰者中「fan 更喜欢」占比越低的方法，更常救下 fan 支持者 → 更偏 fan。\n")

    all_summary_rows = []
    for top_pct in TOP_PCT_LIST:
        res = run_for_top_pct(top_pct, week_div, wr_merged, n_weeks)
        n_top = res["n_top"]
        n_elim_pct = res["n_elim_pct"]
        n_fan_pct = res["n_fan_pct"]
        n_judge_pct = res["n_judge_pct"]
        r_fan_pct = res["r_fan_pct"]
        n_elim_rank = res["n_elim_rank"]
        n_fan_rank = res["n_fan_rank"]
        n_judge_rank = res["n_judge_rank"]
        r_fan_rank = res["r_fan_rank"]

        pct_label = f"前 {top_pct*100:.0f}%"
        print(f"【{pct_label}】 分歧周数 = {n_top}")
        print(f"  百分比法: 淘汰 {n_elim_pct} 人, fan 更喜欢 {int(n_fan_pct)} 人, judge 更喜欢 {int(n_judge_pct)} 人, fan 占比 {r_fan_pct:.2%}")
        print(f"  排名法:   淘汰 {n_elim_rank} 人, fan 更喜欢 {int(n_fan_rank)} 人, judge 更喜欢 {int(n_judge_rank)} 人, fan 占比 {r_fan_rank:.2%}")
        if r_fan_pct < r_fan_rank:
            print("  => 结论: 百分比法更偏 fan")
        elif r_fan_rank < r_fan_pct:
            print("  => 结论: 排名法更偏 fan")
        else:
            print("  => 结论: 两方法 fan 占比相同")
        print()

        all_summary_rows.append({
            "top_pct": top_pct,
            "pct_label": pct_label,
            "n_divergent_weeks": n_top,
            "method": "percentage",
            "n_eliminated": n_elim_pct,
            "n_fan_preferred": int(n_fan_pct),
            "n_judge_preferred": int(n_judge_pct),
            "ratio_fan_preferred": round(r_fan_pct, 4),
        })
        all_summary_rows.append({
            "top_pct": top_pct,
            "pct_label": pct_label,
            "n_divergent_weeks": n_top,
            "method": "ranking",
            "n_eliminated": n_elim_rank,
            "n_fan_preferred": int(n_fan_rank),
            "n_judge_preferred": int(n_judge_rank),
            "ratio_fan_preferred": round(r_fan_rank, 4),
        })

    # 5. 多百分比汇总 CSV
    multi_summary_path = os.path.join(OUT_DIR, "divergent_follow_summary_multi_pct.csv")
    pd.DataFrame(all_summary_rows).to_csv(multi_summary_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 多百分比汇总: {multi_summary_path}")

    # 6. 默认比例：导出分歧周列表、淘汰明细、单比例汇总与报告
    res_default = run_for_top_pct(DEFAULT_TOP_PCT, week_div, wr_merged, n_weeks)
    n_top = res_default["n_top"]
    divergent_weeks = res_default["divergent_weeks"]
    div_only = res_default["div_only"]
    n_elim_pct = res_default["n_elim_pct"]
    n_fan_pct = res_default["n_fan_pct"]
    n_judge_pct = res_default["n_judge_pct"]
    r_fan_pct = res_default["r_fan_pct"]
    n_elim_rank = res_default["n_elim_rank"]
    n_fan_rank = res_default["n_fan_rank"]
    n_judge_rank = res_default["n_judge_rank"]
    r_fan_rank = res_default["r_fan_rank"]

    week_div["is_divergent"] = week_div.apply(
        lambda r: (r["season"], r["week"]) in divergent_weeks, axis=1
    )
    div_weeks_df = week_div[week_div["is_divergent"]][["season", "week", "divergence"]]
    div_weeks_path = os.path.join(OUT_DIR, "divergent_weeks.csv")
    div_weeks_df.to_csv(div_weeks_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 分歧周列表(默认 {DEFAULT_TOP_PCT*100:.0f}%): {div_weeks_path} (共 {n_top} 周)")

    elim_detail = div_only[
        (div_only["elim_pct"] == 1) | (div_only["elim_rank"] == 1)
    ].copy()
    elim_detail["fan_preferred"] = elim_detail["fan_preferred"].astype(int)
    elim_detail["judge_preferred"] = elim_detail["judge_preferred"].astype(int)
    elim_detail_path = os.path.join(OUT_DIR, "divergent_eliminated_detail.csv")
    elim_detail.to_csv(elim_detail_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 分歧周淘汰明细: {elim_detail_path}")

    summary = pd.DataFrame([
        {
            "method": "percentage",
            "n_eliminated": n_elim_pct,
            "n_fan_preferred": int(n_fan_pct),
            "n_judge_preferred": int(n_judge_pct),
            "ratio_fan_preferred": round(r_fan_pct, 4),
        },
        {
            "method": "ranking",
            "n_eliminated": n_elim_rank,
            "n_fan_preferred": int(n_fan_rank),
            "n_judge_preferred": int(n_judge_rank),
            "ratio_fan_preferred": round(r_fan_rank, 4),
        },
    ])
    summary_path = os.path.join(OUT_DIR, "divergent_follow_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[已导出] 汇总(默认 {DEFAULT_TOP_PCT*100:.0f}%): {summary_path}")

    report_path = os.path.join(OUT_DIR, "divergent_follow_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("分歧情形下「跟谁走」判定报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"分歧周定义: 当周 |评委排名-观众排名| 均值 最大的前 {DEFAULT_TOP_PCT*100:.0f}% 周 (共 {n_top} 周)\n\n")
        f.write("淘汰者中「fan 更喜欢」= 观众排名优于评委排名；「judge 更喜欢」= 评委排名优于观众排名。\n")
        f.write("若某方法淘汰者里 fan 更喜欢 占比更低，则该 method 更常「救下 fan 支持者」→ 更偏 fan。\n\n")
        f.write("百分比法: 淘汰 {} 人, 其中 fan 更喜欢 {} 人, judge 更喜欢 {} 人, 占比 {:.2%}\n".format(
            n_elim_pct, int(n_fan_pct), int(n_judge_pct), r_fan_pct
        ))
        f.write("排名法:   淘汰 {} 人, 其中 fan 更喜欢 {} 人, judge 更喜欢 {} 人, 占比 {:.2%}\n\n".format(
            n_elim_rank, int(n_fan_rank), int(n_judge_rank), r_fan_rank
        ))
        if r_fan_pct < r_fan_rank:
            f.write("=> 百分比法淘汰者中 fan 更喜欢占比更低，更常救下 fan 支持者，更偏 fan。\n")
        elif r_fan_rank < r_fan_pct:
            f.write("=> 排名法淘汰者中 fan 更喜欢占比更低，更常救下 fan 支持者，更偏 fan。\n")
        else:
            f.write("=> 两方法淘汰者中 fan 更喜欢占比相同。\n")
    print(f"[已导出] 报告: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
