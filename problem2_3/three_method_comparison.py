"""
问题二（3）：四种方法综合对比

对比四种方法在以下三个指标上的表现：
1. 百分比结合法、排名结合法（直接淘汰末位）
2. 排名+Bottom2+评委裁决、份额+Bottom2+评委裁决

指标：稳定性（Bootstrap 翻转率）、抗操纵性（攻击偏移率）、一致性（Spearman 相关系数）

依赖：需先运行 main.py 生成以下数据文件：
  - weekly_shares.csv
  - percentage_method_rankings.csv
  - ranking_method_rankings.csv
  - new_method_rankings.csv
  - new_method_rankings_pct.csv
  - cleaned_data1.csv（或能提供 placement、is_eliminated 的长表）
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = SCRIPT_DIR

# 数据路径
WEEKLY_SHARES_PATH = DATA_DIR / "weekly_shares.csv"
PERCENTAGE_RANKINGS_PATH = DATA_DIR / "percentage_method_rankings.csv"
RANKING_RANKINGS_PATH = DATA_DIR / "ranking_method_rankings.csv"
NEW_METHOD_RANKINGS_PATH = DATA_DIR / "new_method_rankings.csv"
NEW_METHOD_RANKINGS_PCT_PATH = DATA_DIR / "new_method_rankings_pct.csv"
CLEANED_LONG_PATH = DATA_DIR / "cleaned_data1.csv"


def build_long_df_for_evaluation():
    """从 main.py 导出的数据文件合并构建评估用 long_df"""
    import pandas as pd

    # 基准：new_method_rankings 含 judge_rank, audience_rank, final_rank_alt, judge_total_score
    base = pd.read_csv(NEW_METHOD_RANKINGS_PATH)

    # 合并 weekly_shares
    shares = pd.read_csv(WEEKLY_SHARES_PATH)
    df = base.merge(
        shares[["season", "week", "celebrity_name", "judge_share", "audience_share", "combined_share"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )

    # 合并 份额+Bottom2 的 final_rank_alt_pct
    if NEW_METHOD_RANKINGS_PCT_PATH.exists():
        new_pct = pd.read_csv(NEW_METHOD_RANKINGS_PCT_PATH)
        df = df.merge(
            new_pct[["season", "week", "celebrity_name", "final_rank_alt_pct"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )

    # 合并 percentage 的 combined_share_rank
    pct = pd.read_csv(PERCENTAGE_RANKINGS_PATH)
    pct = pct.rename(columns={"final_rank": "combined_share_rank"})
    df = df.merge(
        pct[["season", "week", "celebrity_name", "combined_share_rank"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )

    # 合并 ranking 的 combined_rank_final
    rank_df = pd.read_csv(RANKING_RANKINGS_PATH)
    rank_df = rank_df.rename(columns={"final_rank": "combined_rank_final"})
    df = df.merge(
        rank_df[["season", "week", "celebrity_name", "combined_rank_final"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )

    # 合并 cleaned_data1 的 placement, is_eliminated, n_eliminated
    if CLEANED_LONG_PATH.exists():
        cleaned = pd.read_csv(CLEANED_LONG_PATH)
        needed = ["season", "week", "celebrity_name", "placement", "exit_week", "is_eliminated", "n_eliminated"]
        if all(c in cleaned.columns for c in needed):
            df = df.merge(
                cleaned[needed].drop_duplicates(subset=["season", "week", "celebrity_name"]),
                on=["season", "week", "celebrity_name"],
                how="left",
            )
        else:
            # 若列名不同，尝试用 is_eliminated 从 base 推断（new 法无 is_eliminated，需从别处来）
            pass
    else:
        # 若无 cleaned_data1，用 pred_eliminated_alt 近似 is_eliminated（实际淘汰来自历史数据）
        df["is_eliminated"] = df.get("pred_eliminated_alt", 0)
        df["placement"] = None
        df["n_eliminated"] = df.groupby(["season", "week"])["is_eliminated"].transform("sum")

    # 确保 combined_rank 存在（judge_rank + audience_rank）
    if "combined_rank" not in df.columns:
        df["combined_rank"] = df["judge_rank"] + df["audience_rank"]

    return df


def main():
    print("=" * 60)
    print("问题二（3）：四种方法综合对比")
    print("=" * 60)

    # 检查数据文件
    required = [
        WEEKLY_SHARES_PATH,
        PERCENTAGE_RANKINGS_PATH,
        RANKING_RANKINGS_PATH,
        NEW_METHOD_RANKINGS_PATH,
        NEW_METHOD_RANKINGS_PCT_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\n[错误] 缺少数据文件，请先运行 main.py：")
        for p in missing:
            print(f"  - {p.name}")
        return

    # 构建 long_df
    print("\n[1] 加载并合并数据...")
    long_df = build_long_df_for_evaluation()

    # 若 cleaned_data1 不存在，is_eliminated 需从实际淘汰结果来
    if "is_eliminated" not in long_df.columns or long_df["is_eliminated"].isna().all():
        print("[警告] 缺少 is_eliminated，将使用 pred_eliminated_alt 作为近似")
        long_df["is_eliminated"] = long_df.get("pred_eliminated_alt", 0)
    if "placement" not in long_df.columns or long_df["placement"].isna().all():
        print("[警告] 缺少 placement，一致性指标可能不准确")

    print(f"  合并后记录数: {len(long_df)}")

    # 调用综合评估
    print("\n[2] 计算四种方法的稳定性、抗操纵性、一致性...")
    from evaluation.comprehensive_metrics import comprehensive_evaluation

    result = comprehensive_evaluation(long_df, output_dir=str(OUT_DIR))

    print(f"\n[完成] 结果已导出至: {OUT_DIR / 'metric_comprehensive_comparison.csv'}")


if __name__ == "__main__":
    main()
