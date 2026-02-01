"""
问题三第二小问：评委排名 vs 观众排名 —— 职业舞者与名人特征的影响是否一致？

样本单位：(celebrity, season, week)，每人每季每周一条观测，不聚合。

因变量（周内归一化排名，0=当周最好、1=当周最差，跨周可比）：
  - y1 = judge_rank_norm：由当周评委总分周内排序得到 judge_rank，再 (rank-1)/(n-1)
  - y2 = audience_rank_norm：观众解耦按赛季类型：3–27 季用份额法（和=2）→ audience_share 再排名；
    1–2 季与 28 季以后用排名法 n(n+1) → combined_rank_score - judge_rank 再排名。再 (rank-1)/(n-1)
  说明：audience_rank 为估计，非官方真实票数。

固定效应：season-week（同一季同一周一个 FE），吸收该周共同环境。
聚类标准误：cluster by (celebrity, season)。
自变量 X：与 Q3.1 同套（年龄组、国家、行业、职业舞者）。
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from preprocess.load_and_clean import load_data, clean_data, reshape_to_long_weeks
from preprocess.handle_special_cases import mark_withdrawal_and_elimination
from features.judge_features import build_judge_features
from features.environment_features import build_environment_features
from features.personal_features import build_personal_features
from models.vote_share_model import EliminationProbModel

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# 与 main.py 一致的淘汰模型特征列
FEATURE_COLS_ELIM = [
    "judge_mean", "judge_std", "judge_mean_roll_std", "judge_std_roll_std",
    "env_mean", "env_std", "env_count", "env_special", "env_week", "env_intensity",
    "env_boundary_margin", "env_ties", "env_mean_roll_std", "env_std_roll_std",
    "env_count_delta", "env_intensity_delta", "env_ties_delta", "env_boundary_margin_delta",
    "weeks_in_competition", "rank_max_change", "rank_mean", "rank_std",
    "rank_spike_count", "bottom_k_count",
]


def build_long_with_ranks() -> pd.DataFrame:
    """复现 main.py 流程：得到含 judge_rank、audience_rank 的 (celebrity, season, week) 长表。"""
    path = project_root / "data" / "2026_MCM_Problem_C_Data.csv"
    raw = load_data(str(path))
    if raw is None:
        raise FileNotFoundError(f"数据不存在: {path}")
    cleaned = clean_data(raw)
    cleaned = mark_withdrawal_and_elimination(cleaned)
    long_df = reshape_to_long_weeks(cleaned)

    max_week = long_df.groupby("season")["week"].max().reset_index(name="max_week")
    long_df = long_df.merge(max_week, on="season", how="left")
    is_champion = (long_df["placement"] == 1)
    long_df.loc[is_champion, "exit_week"] = 0
    long_df["is_eliminated"] = (long_df["week"] == long_df["exit_week"]).astype(int)
    long_df = long_df[long_df["judge_total_score"] > 0].copy()

    elim_counts = long_df.groupby(["season", "week"])["is_eliminated"].sum().reset_index(name="n_eliminated")
    long_df = long_df.merge(elim_counts, on=["season", "week"], how="left")
    long_df["n_eliminated"] = long_df["n_eliminated"].fillna(0).astype(int)

    long_df = build_judge_features(long_df)
    long_df = build_environment_features(long_df)
    long_df = build_personal_features(long_df)
    long_df[FEATURE_COLS_ELIM] = long_df[FEATURE_COLS_ELIM].fillna(0)

    train_mask = (long_df["season"] >= 1) & (long_df["season"] <= 34)
    elim_model = EliminationProbModel()
    elim_model.fit(
        long_df.loc[train_mask, FEATURE_COLS_ELIM],
        long_df.loc[train_mask, "is_eliminated"],
    )
    long_df["elimination_prob"] = elim_model.predict(long_df[FEATURE_COLS_ELIM]).clip(0, 1)

    long_df["judge_share"] = long_df.groupby(["season", "week"])["judge_total_score"].transform(
        lambda s: s / (s.sum() + 1e-10)
    )
    long_df["survival_prob"] = 1 - long_df["elimination_prob"]
    long_df["judge_rank"] = long_df.groupby(["season", "week"])["judge_total_score"].rank(
        ascending=False, method="min"
    )

    # 按赛季类型解耦观众：3–27 份额法（和=2），1–2 与 28+ 排名法 n(n+1)
    mask_share = (long_df["season"] >= 3) & (long_df["season"] <= 27)
    mask_rank = (long_df["season"] <= 2) | (long_df["season"] >= 28)

    # 份额法（3–27）：estimated_combined_share 和=2 → audience_share → 按份额排名
    long_df["estimated_combined_share"] = long_df.groupby(["season", "week"])["survival_prob"].transform(
        lambda s: 2.0 * s / (s.sum() + 1e-10)
    )
    long_df["audience_share_raw"] = long_df["estimated_combined_share"] - long_df["judge_share"]

    def norm_share(raw_series):
        raw = raw_series.values
        shifted = raw - raw.min()
        total = shifted.sum()
        return shifted / total if total > 1e-10 else np.ones(len(raw)) / len(raw)

    long_df["audience_share"] = np.nan
    long_df.loc[mask_share, "audience_share"] = long_df.loc[mask_share].groupby(["season", "week"])["audience_share_raw"].transform(norm_share)

    # 排名法（1–2, 28+）：combined_rank_score = norm(elim_prob)*n*(n+1)，audience_rank_score = combined_rank_score - judge_rank
    def calc_combined_rank_score(group):
        elim_prob = group["elimination_prob"].values
        n = len(group)
        norm = elim_prob / (elim_prob.sum() + 1e-10)
        return norm * n * (n + 1)

    combined_rank_score = long_df.groupby(["season", "week"]).apply(
        lambda g: pd.Series(calc_combined_rank_score(g), index=g.index)
    ).reset_index(level=[0, 1], drop=True)
    long_df["audience_rank_score"] = np.nan
    long_df.loc[mask_rank, "audience_rank_score"] = combined_rank_score.loc[mask_rank] - long_df.loc[mask_rank, "judge_rank"]

    # 观众排名：份额季按 audience_share 排，排名季按 audience_rank_score 排
    long_df["audience_rank"] = np.nan
    long_df.loc[mask_share, "audience_rank"] = long_df.loc[mask_share].groupby(["season", "week"])["audience_share"].rank(
        ascending=False, method="min"
    )
    long_df.loc[mask_rank, "audience_rank"] = long_df.loc[mask_rank].groupby(["season", "week"])["audience_rank_score"].rank(
        ascending=True, method="min"
    )

    # 排名季补 audience_share：2*(n+1-rank)/(n*(n+1)) 再按周归一化（有并列时周内和才为 1）
    n_sw = long_df.groupby(["season", "week"])["judge_rank"].transform("size")
    raw_share_rank = 2.0 * (n_sw + 1 - long_df["audience_rank"]) / (n_sw * (n_sw + 1))
    rank_subset = long_df.loc[mask_rank, ["season", "week"]].copy()
    rank_subset["_raw"] = raw_share_rank.loc[mask_rank]
    rank_subset["_sum"] = rank_subset.groupby(["season", "week"])["_raw"].transform("sum")
    long_df.loc[mask_rank, "audience_share"] = rank_subset["_raw"] / rank_subset["_sum"]

    return long_df


def rank_to_norm(rank_series: pd.Series, n_by_group: pd.Series) -> pd.Series:
    """周内归一化排名: (rank - 1) / (n - 1)，0=最好，1=最差。n=1 时置 0。"""
    n = n_by_group
    denom = n - 1
    denom = denom.replace(0, 1)  # 避免除零，该周仅1人时归一化为0
    return (rank_series - 1) / denom


def prepare_weekly_features(long_df: pd.DataFrame):
    """
    在 (celebrity, season, week) 长表上构造：
    - judge_rank_norm, audience_rank_norm
    - season_week（用于 FE）
    - 年龄组、国家、行业、职业舞者哑变量（与 placement_model 同逻辑）
    - cluster_id = (celebrity_name, season)
    返回 (df, y1, y2, X, X_names, cluster_ids, ref_names)。
    ref_names: 各因素被 drop_first 掉的水平名，用于输出时补全为 β=0，做到“全都给出 beta”。
    """
    df = long_df.copy()
    n_sw = df.groupby(["season", "week"])["judge_rank"].transform("size")
    df["judge_rank_norm"] = rank_to_norm(df["judge_rank"], n_sw)
    df["audience_rank_norm"] = rank_to_norm(df["audience_rank"], n_sw)
    df["season_week"] = "s" + df["season"].astype(str) + "_w" + df["week"].astype(str)
    df["cluster_id"] = df["celebrity_name"].astype(str) + "_" + df["season"].astype(str)

    # 年龄组
    age = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    age = age.fillna(age.median() if age.notna().any() else 30)
    bins, labels = [0, 30, 40, 50, 150], ["<30", "30-39", "40-49", "50+"]
    df["age_group"] = pd.cut(age, bins=bins, labels=labels, right=False)

    # 行业 Top8 + Other（按人-季一条计，避免存活久的人权重大）
    base = df.drop_duplicates(["season", "celebrity_name"])
    ind_series = base["celebrity_industry"].fillna("Unknown").astype(str)
    top_ind = ind_series.value_counts().head(8).index.tolist()
    ind = df["celebrity_industry"].fillna("Unknown").astype(str)
    df["ind_g"] = ind.apply(lambda x: x if x in top_ind else "Other")

    # 国家 Top5 + Other（按人-季一条计；仅用 country/region 列，不把州当国家）
    if "celebrity_homecountry/region" in df.columns:
        cty_series = base["celebrity_homecountry/region"].fillna("Unknown").astype(str)
        top_cty = cty_series.value_counts().head(5).index.tolist()
        cty = df["celebrity_homecountry/region"].fillna("Unknown").astype(str)
    else:
        top_cty = []
        cty = pd.Series("Unknown", index=df.index)
    df["cty_g"] = cty.apply(lambda x: x if x in top_cty else "Other_Country")

    # 职业舞者：跨赛季出现>=2次单独编码（使用 build_personal_features 已生成的 partner）
    if "partner" not in df.columns:
        df["partner"] = df["ballroom_partner"].fillna("Unknown").astype(str)
    p_seasons = df.groupby("partner")["season"].nunique()
    top_pro = p_seasons[p_seasons >= 2].index.tolist()
    df["pro_g"] = df["partner"].apply(lambda x: x if x in top_pro else "Other_Partner")

    # 哑变量：season_week drop_first，年龄/国家/行业/职业舞者 drop_first
    sw_dum = pd.get_dummies(df["season_week"], prefix="sw", drop_first=True)
    age_dum = pd.get_dummies(df["age_group"], prefix="age", drop_first=True)
    cty_dum = pd.get_dummies(df["cty_g"], prefix="cty", drop_first=True)
    ind_dum = pd.get_dummies(df["ind_g"], prefix="ind", drop_first=True)
    pro_dum = pd.get_dummies(df["pro_g"], prefix="pro", drop_first=True)

    X_parts = [sw_dum, age_dum, cty_dum, ind_dum, pro_dum]
    X_names = (
        list(sw_dum.columns) + list(age_dum.columns) + list(cty_dum.columns)
        + list(ind_dum.columns) + list(pro_dum.columns)
    )
    X = np.column_stack([p.astype(np.float64).fillna(0).values for p in X_parts])
    y1 = df["judge_rank_norm"].values
    y2 = df["audience_rank_norm"].values
    # 聚类需要整数组标识
    cluster_ids, _ = pd.factorize(df["cluster_id"])
    # 被 drop_first 掉的水平（用于输出时补全为 β=0）
    ref_names = {
        "age": "age_" + str(labels[0]),
        "cty": "cty_" + sorted(df["cty_g"].unique())[0],
        "ind": "ind_" + sorted(df["ind_g"].unique())[0],
        "pro": "pro_" + sorted(df["pro_g"].unique())[0],
    }
    return df, y1, y2, X, X_names, cluster_ids, ref_names


def expand_coef_with_ref(res_core: pd.DataFrame, ref_names: dict) -> pd.DataFrame:
    """
    在系数表中为每个因素的“参照水平”补一行 β=0、pvalue=NaN，使输出为“全都给出 beta”。
    各因素内顺序：参照水平（0） + 其余水平（与 res_core 中一致）。若有 se 列则参照行 se=NaN。
    """
    rows = []
    ref_row = {"coef": 0.0, "pvalue": np.nan}
    if "se" in res_core.columns:
        ref_row["se"] = np.nan
    for prefix, ref_var in [("age_", ref_names["age"]), ("cty_", ref_names["cty"]), ("ind_", ref_names["ind"]), ("pro_", ref_names["pro"])]:
        block = [i for i in res_core.index if str(i).startswith(prefix)]
        if not block:
            continue
        rows.append(pd.DataFrame(ref_row, index=[ref_var]))
        rows.append(res_core.loc[block])
    if not rows:
        return res_core
    return pd.concat(rows, axis=0)


def ols_cluster(y: np.ndarray, X: np.ndarray, names: list, cluster_ids: np.ndarray):
    """OLS + 按 cluster_ids 聚类的稳健标准误。返回 (coef_df, r2, adj_r2)。"""
    n, k = X.shape
    X_const = np.column_stack([np.ones(n), X])
    all_names = ["const"] + names
    if not HAS_SM:
        # 简单 OLS 无聚类
        XtX = X_const.T @ X_const
        Xty = X_const.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
        resid = y - X_const @ beta
        ss_res = resid @ resid
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10) if n > 1 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
        mse = ss_res / max(n - k - 1, 1)
        try:
            se = np.sqrt(mse * np.diag(np.linalg.inv(XtX)))[1:]
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
        try:
            from scipy import stats as spst
            df_resid = max(n - k - 1, 1)
            t = beta[1:] / (se + 1e-10)
            pval = 2 * (1 - spst.t.cdf(np.abs(t), df_resid))
        except Exception:
            pval = np.full(k, np.nan)
        coef_df = pd.DataFrame({"coef": beta[1:], "pvalue": pval}, index=names)
        return coef_df, r2, adj_r2

    res = sm.OLS(y, X_const).fit()
    r2, adj_r2 = res.rsquared, res.rsquared_adj
    try:
        robust = res.get_robustcov_results(cov_type="cluster", groups=cluster_ids)
        coef = robust.params[1:]
        pval = robust.pvalues[1:]
        bse = robust.bse[1:]
    except Exception:
        coef = res.params[1:]
        pval = res.pvalues[1:]
        bse = res.bse[1:]
    coef_df = pd.DataFrame({"coef": coef, "se": bse, "pvalue": pval}, index=names)
    return coef_df, r2, adj_r2


def run_interaction_test(
    y1: np.ndarray, y2: np.ndarray, X: np.ndarray, X_names: list, cluster_ids: np.ndarray, out_dir: Path
) -> None:
    """
    堆叠 judge/audience，加 type（观众=1）与 X*type 交互；交互项显著表示该因素对评委/观众影响方式不同。
    仅对非 season_week 的 X 做交互。
    """
    n = len(y1)
    sw_idx = [i for i, name in enumerate(X_names) if str(name).startswith("sw_")]
    core_idx = [i for i, name in enumerate(X_names) if not str(name).startswith("sw_")]
    if not core_idx:
        print("    [交互检验] 无核心变量，跳过")
        return
    X_sw = X[:, sw_idx]
    X_core = X[:, core_idx]
    core_names = [X_names[i] for i in core_idx]
    type_col = np.concatenate([np.zeros(n), np.ones(n)])
    X_core_x_type = np.vstack([np.zeros_like(X_core), X_core])
    X_stacked = np.column_stack([
        np.vstack([X_sw, X_sw]),
        np.vstack([X_core, X_core]),
        type_col[:, None],
        X_core_x_type,
    ])
    inter_names = [c + "_x_audience" for c in core_names]
    stacked_names = [X_names[i] for i in sw_idx] + core_names + ["type_audience"] + inter_names
    y_stacked = np.concatenate([y1, y2])
    cluster_stacked = np.concatenate([cluster_ids, cluster_ids])

    if not HAS_SM:
        print("    [交互检验] 需要 statsmodels，跳过")
        return
    X_const = np.column_stack([np.ones(len(y_stacked)), X_stacked])
    res = sm.OLS(y_stacked, X_const).fit()
    try:
        robust = res.get_robustcov_results(cov_type="cluster", groups=cluster_stacked)
        inter_coef = robust.params[1:][-len(inter_names):]
        inter_pval = robust.pvalues[1:][-len(inter_names):]
    except Exception:
        inter_coef = res.params[1:][-len(inter_names):]
        inter_pval = res.pvalues[1:][-len(inter_names):]
    inter_df = pd.DataFrame({"coef": inter_coef, "pvalue": inter_pval}, index=inter_names)
    inter_df.to_csv(out_dir / "interaction_judge_vs_audience.csv", encoding="utf-8-sig")
    n_sig = (inter_pval < 0.05).sum()
    print(f"    [交互检验] X*type_audience 显著数: {n_sig}/{len(inter_names)} → 见 interaction_judge_vs_audience.csv")


def _en_label(name: str) -> str:
    """变量名转简短英文标签（用于图）。"""
    s = str(name)
    if s.startswith("age_"):
        return s.replace("age_", "Age ")
    if s.startswith("cty_"):
        return s.replace("cty_", "")
    if s.startswith("ind_"):
        return s.replace("ind_", "")
    if s.startswith("pro_"):
        return s.replace("pro_", "")
    return s


def plot_comparison(res_judge: pd.DataFrame, res_audience: pd.DataFrame, out_path: Path):
    """Judge vs audience: vertical bars — x=variable, y=coefficient (English)."""
    if not HAS_PLOT:
        return
    def core_vars(idx):
        return [i for i in idx if not str(i).startswith("sw_")]

    j_idx = core_vars(res_judge.index)
    a_idx = core_vars(res_audience.index)
    common = sorted(set(j_idx) & set(a_idx), key=lambda x: (x.split("_")[0], x))
    if not common:
        return
    j_coef = res_judge.loc[common, "coef"].values
    a_coef = res_audience.loc[common, "coef"].values
    j_p = res_judge.loc[common, "pvalue"].values
    a_p = res_audience.loc[common, "pvalue"].values
    labels = [_en_label(c) for c in common]

    # 纵向图：x=变量，y=系数，条竖着
    n = len(common)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.32), 6))
    x_pos = np.arange(n)
    bar_w = 0.36
    ax.axhline(0, color="gray", linewidth=0.8, zorder=0)
    ax.bar(x_pos - bar_w / 2, j_coef, width=bar_w, color="#2c3e50", alpha=0.9, label="Judge rank", zorder=2)
    ax.bar(x_pos + bar_w / 2, a_coef, width=bar_w, color="#e67e22", alpha=0.9, label="Audience rank (est.)", zorder=2)
    for i in range(n):
        j_sig = "*" if j_p[i] < 0.05 else ""
        a_sig = "*" if a_p[i] < 0.05 else ""
        off = 0.02
        ax.text(x_pos[i] - bar_w / 2, j_coef[i] + off if j_coef[i] >= 0 else j_coef[i] - off, f"{j_coef[i]:+.2f}{j_sig}",
                va="bottom" if j_coef[i] >= 0 else "top", ha="center", fontsize=6, rotation=90)
        ax.text(x_pos[i] + bar_w / 2, a_coef[i] + off if a_coef[i] >= 0 else a_coef[i] - off, f"{a_coef[i]:+.2f}{a_sig}",
                va="bottom" if a_coef[i] >= 0 else "top", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set_ylabel("Coefficient \u03b2 (positive = worse rank that week)", fontsize=10)
    ax.set_title("Judge vs Audience: Effect on Weekly Normalized Rank (* p<0.05; cluster by celebrity-season)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  对比图已保存: {out_path}")


def main():
    out_dir = project_root / "problem3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题三第二小问：评委排名 vs 观众排名 —— 特征影响是否一致？")
    print("=" * 60)
    print("样本：(celebrity, season, week)，不聚合。")
    print("因变量：周内归一化排名 rank_norm = (rank-1)/(n-1)，0=最好，1=最差。")
    print("  y1 = judge_rank_norm（评委周内排名，原始数据）")
    print("  y2 = audience_rank_norm（观众周内排名，由第一问估计的观众份额/排名得到，非官方真实票数）")
    print("固定效应：season-week。聚类：(celebrity, season)。")
    print()

    print("[1] 构建周粒度数据与评委/观众排名...")
    long_df = build_long_with_ranks()
    print(f"    观测数: {len(long_df)} (人-季-周)")

    print("[2] 周内归一化排名与特征...")
    df, y1, y2, X, X_names, cluster_ids, ref_names = prepare_weekly_features(long_df)
    print(f"    judge_rank_norm 范围: {y1.min():.4f} ~ {y1.max():.4f}")
    print(f"    audience_rank_norm 范围: {y2.min():.4f} ~ {y2.max():.4f}")
    print(f"    自变量维度: {X.shape[1]} (含 season-week FE)")

    print("\n[3] 模型 1: judge_rank_norm ~ X + season-week FE，cluster (celebrity, season)")
    res_judge, r2_j, adj_r2_j = ols_cluster(y1, X, X_names, cluster_ids)
    print(f"    R² = {r2_j:.4f}   Adj R² = {adj_r2_j:.4f}")

    print("\n[4] 模型 2: audience_rank_norm ~ X + season-week FE，cluster (celebrity, season)")
    res_audience, r2_a, adj_r2_a = ols_cluster(y2, X, X_names, cluster_ids)
    print(f"    R² = {r2_a:.4f}   Adj R² = {adj_r2_a:.4f}")

    # 系数表：只保留非 season_week 的变量，并补全参照水平为 β=0，做到全都给出 beta
    core_idx = [i for i in res_judge.index if not str(i).startswith("sw_")]
    res_judge_core = res_judge.loc[core_idx]
    res_audience_core = res_audience.loc[core_idx]
    res_judge_expanded = expand_coef_with_ref(res_judge_core, ref_names)
    res_audience_expanded = expand_coef_with_ref(res_audience_core, ref_names)
    res_judge_expanded[["coef"]].to_csv(out_dir / "coef_judge_rank_norm.csv", encoding="utf-8-sig")
    res_audience_expanded[["coef"]].to_csv(out_dir / "coef_audience_rank_norm.csv", encoding="utf-8-sig")
    print(f"\n[5] 系数表已保存（含全部 β，基准水平 β=0）: coef_judge_rank_norm.csv, coef_audience_rank_norm.csv")

    plot_comparison(res_judge_expanded, res_audience_expanded, out_dir / "judge_vs_audience_effects.png")

    print("\n[6] 交互检验：堆叠 judge/audience + X*type_audience，交互显著即影响方式不同")
    run_interaction_test(y1, y2, X, X_names, cluster_ids, out_dir)

    print("\n[7] 小结：系数解释 —— 正系数 = 当周排名更靠后（更差）。比较两表可知评委与观众端影响是否同向、是否均显著；交互表可知哪些因素在评委/观众端作用不同。")
    print("[完成]")


if __name__ == "__main__":
    main()
