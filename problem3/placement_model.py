"""
问题三：职业舞者与名人特征对最终排名的影响

目标：分析各种职业舞者 + 名人静态特征（年龄、行业等）对名人在比赛中的
      最终表现（最终排名 placement）的影响有多大。题目中这些因素为同级别，
      不做“先名人特征、再职业舞者”的嵌套比较。

做法：
  - 因变量：placement（最终排名，数值越小越好）
  - 一个模型：placement ~ 赛季(控制) + 年龄组 + 国家 + 行业 + 职业舞者

不依赖淘汰模型、粉丝份额、early_fan/early_judge，仅用原始数据。
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


def build_celeb_season():
    """构建 celebrity-season 数据（每人每季一行），含 placement、年龄、行业、国家、舞伴。"""
    path = project_root / "data" / "2026_MCM_Problem_C_Data.csv"
    raw = load_data(str(path))
    if raw is None:
        raise FileNotFoundError(f"数据不存在: {path}")
    cleaned = clean_data(raw)
    cleaned = mark_withdrawal_and_elimination(cleaned)
    long_df = reshape_to_long_weeks(cleaned)
    long_df = long_df.sort_values(["season", "celebrity_name", "week"])
    cs = long_df.groupby(["season", "celebrity_name"], as_index=False).first()
    cs = cs[cs["placement"].notna() & (cs["placement"] > 0)].copy()
    return cs


def prepare_features(cs: pd.DataFrame, use_normalized_placement: bool = False):
    """
    准备嵌套模型用的特征块与因变量。
    use_normalized_placement: 若 True，y 用赛季内按 min-max 归一化排名 (placement-min)/(max-min)，使跨赛季尺度一致。
    返回: y, X_season, X_celeb, X_pro, names_season, names_celeb, names_pro
    """
    df = cs.reset_index(drop=True)
    n = len(df)

    # 年龄分组（分类变量，参照组 <30 岁）
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median() if df["age"].notna().any() else 30)
    bins = [0, 30, 40, 50, 150]
    labels = ["<30", "30-39", "40-49", "50+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # 行业：Top 8 + Other
    df["industry"] = df["celebrity_industry"].fillna("Unknown").astype(str)
    top_ind = df["industry"].value_counts().head(8).index.tolist()
    df["ind_g"] = df["industry"].apply(lambda x: x if x in top_ind else "Other")

    # 国家：Top 5 + Other
    df["country"] = df["celebrity_homecountry/region"].fillna("Unknown").astype(str)
    top_cty = df["country"].value_counts().head(5).index.tolist()
    df["cty_g"] = df["country"].apply(lambda x: x if x in top_cty else "Other_Country")

    # 职业舞者：跨赛季出现 >= 2 次单独编码
    df["partner"] = df["ballroom_partner"].fillna("Unknown").astype(str)
    p_seasons = df.groupby("partner")["season"].nunique()
    top_pro = p_seasons[p_seasons >= 2].index.tolist()
    df["pro_g"] = df["partner"].apply(lambda x: x if x in top_pro else "Other_Partner")

    if use_normalized_placement:
        # 赛季内按 min-max 归一化：(placement-min)/(max-min)，跨赛季尺度一致（某季 min≠1 时也成立）
        by_season = df.groupby("season")["placement"]
        p_max = by_season.transform("max")
        p_min = by_season.transform("min")
        y = ((df["placement"].astype(float) - p_min) / (p_max - p_min + 1e-10)).values
    else:
        y = df["placement"].astype(float).values

    sea_dum = pd.get_dummies(df["season"].astype(str), prefix="s", drop_first=True)
    sea_dum.reset_index(drop=True, inplace=True)
    X_season = sea_dum.astype(np.float64).fillna(0).values
    names_season = list(sea_dum.columns)

    age_dum = pd.get_dummies(df["age_group"], prefix="age", drop_first=True).reset_index(drop=True)
    cty_dum = pd.get_dummies(df["cty_g"], prefix="cty", drop_first=True).reset_index(drop=True)
    ind_dum = pd.get_dummies(df["ind_g"], prefix="ind", drop_first=True).reset_index(drop=True)
    X_celeb = np.column_stack([
        age_dum.astype(np.float64).fillna(0).values,
        cty_dum.astype(np.float64).fillna(0).values,
        ind_dum.astype(np.float64).fillna(0).values,
    ])
    names_celeb = list(age_dum.columns) + list(cty_dum.columns) + list(ind_dum.columns)

    pro_dum = pd.get_dummies(df["pro_g"], prefix="pro", drop_first=True).reset_index(drop=True)
    X_pro = pro_dum.astype(np.float64).fillna(0).values
    names_pro = list(pro_dum.columns)

    return y, X_season, X_celeb, X_pro, names_season, names_celeb, names_pro


def ols_fit(X: np.ndarray, y: np.ndarray, names: list):
    """OLS 回归，返回 (系数 DataFrame, R², Adjusted R²)。"""
    n, k = X.shape
    X_const = np.column_stack([np.ones(n), X])
    if HAS_SM:
        res = sm.OLS(y, X_const).fit()
        coef = res.params[1:]
        pval = res.pvalues[1:]
        r2 = res.rsquared
        adj_r2 = res.rsquared_adj
    else:
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
        coef = beta[1:]
        t = coef / (se + 1e-10)
        try:
            from scipy import stats as spst
            pval = 2 * (1 - spst.t.cdf(np.abs(t), max(n - k - 1, 1)))
        except ImportError:
            pval = np.full(k, np.nan)
    out = pd.DataFrame({"coef": coef, "pvalue": pval}, index=names)
    return out, r2, adj_r2


def plot_placement_effects(res_core: pd.DataFrame, out_path: Path):
    """
    一张图展示各因素对最终排名的影响：年龄、国家、行业、职业舞者（系数条形图）。
    负系数=排名更靠前，正系数=排名更靠后。
    """
    if not HAS_PLOT:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("各因素对最终排名的影响（β：负=更靠前，正=更靠后）", fontsize=12)

    def bar_panel(ax, names: list, coefs: np.ndarray, pvals: np.ndarray, title: str, colors=None):
        if not names:
            ax.set_visible(False)
            return
        y_pos = np.arange(len(names))[::-1]
        bars = ax.barh(y_pos, coefs, color=colors, edgecolor="gray", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("系数 β")
        ax.set_title(title)
        for i, (c, p) in enumerate(zip(coefs, pvals)):
            sig = "*" if p < 0.05 else ""
            ax.text(c + (0.1 if c >= 0 else -0.1), y_pos[i], f"{c:+.2f}{sig}", va="center", ha="left" if c >= 0 else "right", fontsize=7)
        xlo = min(coefs.min() - 0.3, -0.3)
        xhi = max(coefs.max() + 0.3, 0.3)
        ax.set_xlim(xlo, xhi)

    # 年龄组（参照：<30 岁）
    age_vars = [i for i in res_core.index if i.startswith("age_")]
    if age_vars:
        names = [i.replace("age_", "") for i in age_vars]
        coefs = res_core.loc[age_vars, "coef"].values
        pvals = res_core.loc[age_vars, "pvalue"].values
        colors = ["#2ecc71" if c < 0 else "#e74c3c" for c in coefs]
        bar_panel(axes[0, 0], names, coefs, pvals, "年龄组（参照：<30岁）", colors)

    # 国家
    cty_vars = [i for i in res_core.index if i.startswith("cty_")]
    if cty_vars:
        names = [i.replace("cty_", "") for i in cty_vars]
        coefs = res_core.loc[cty_vars, "coef"].values
        pvals = res_core.loc[cty_vars, "pvalue"].values
        colors = ["#2ecc71" if c < 0 else "#e74c3c" for c in coefs]
        bar_panel(axes[0, 1], names, coefs, pvals, "国家（相对参照组）", colors)

    # 行业
    ind_vars = [i for i in res_core.index if i.startswith("ind_")]
    if ind_vars:
        names = [i.replace("ind_", "") for i in ind_vars]
        coefs = res_core.loc[ind_vars, "coef"].values
        pvals = res_core.loc[ind_vars, "pvalue"].values
        colors = ["#2ecc71" if c < 0 else "#e74c3c" for c in coefs]
        bar_panel(axes[1, 0], names, coefs, pvals, "行业（相对参照组）", colors)

    # 职业舞者：按 |coef| 取前 15
    pro_vars = [i for i in res_core.index if i.startswith("pro_")]
    if pro_vars:
        order = sorted(pro_vars, key=lambda x: abs(res_core.loc[x, "coef"]), reverse=True)[:15]
        names = [i.replace("pro_", "") for i in order]
        coefs = res_core.loc[order, "coef"].values
        pvals = res_core.loc[order, "pvalue"].values
        colors = ["#2ecc71" if c < 0 else "#e74c3c" for c in coefs]
        bar_panel(axes[1, 1], names, coefs, pvals, "职业舞者（前15 |β|，相对参照组）", colors)

    fig.text(0.5, 0.01, "绿色=排名更靠前  红色=排名更靠后  * p<0.05", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表已保存: {out_path}")


def main():
    out_dir = project_root / "problem3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题三：职业舞者与名人特征对最终排名的影响")
    print("=" * 60)
    print("因变量：placement（最终排名，越小越好）")
    print("一个模型：赛季(控制) + 年龄组 + 国家 + 行业 + 职业舞者（同级别）")
    print("不依赖粉丝份额或淘汰模型。\n")

    cs = build_celeb_season()
    print(f"[1] 样本量: {len(cs)} (人-季)")

    y, X_season, X_celeb, X_pro, names_season, names_celeb, names_pro = prepare_features(cs)
    print(f"    placement 范围: {y.min():.2f} ~ {y.max():.2f}")

    # 一个模型：赛季 + 年龄 + 国家 + 行业 + 职业舞者（同级别）
    print("\n[2] 模型拟合：placement ~ 赛季 + 年龄组 + 国家 + 行业 + 职业舞者")
    print("-" * 55)
    X_full = np.column_stack([X_season, X_celeb, X_pro])
    names_full = names_season + names_celeb + names_pro
    res_full, r2, adj_r2 = ols_fit(X_full, y, names_full)
    print(f"  R² = {r2:.4f}   Adjusted R² = {adj_r2:.4f}")

    # 稳健性检验：赛季内归一化排名（跨赛季尺度一致）
    print("\n[2b] 稳健性检验：因变量 = 赛季内按 min-max 归一化排名 (placement-min)/(max-min)")
    print("-" * 55)
    y_norm, _, _, _, _, _, _ = prepare_features(cs, use_normalized_placement=True)
    _, _, adj_r2_n = ols_fit(X_full, y_norm, names_full)
    print(f"  Adjusted R² = {adj_r2_n:.4f}")

    # 系数表（只保留年龄、国家、行业、职业舞者，不报赛季）
    core_idx = [i for i in res_full.index if not i.startswith("s_")]
    res_core = res_full.loc[core_idx]
    res_core.to_csv(out_dir / "coef_placement_only.csv", encoding="utf-8-sig")
    plot_placement_effects(res_core, out_dir / "placement_effects.png")

    print("\n[3] 各因素对最终排名的影响（系数：负=排名更靠前，*=p<0.05）")
    print("-" * 50)
    for prefix, label in [("age_", "年龄组（参照：<30岁）"), ("cty_", "国家"), ("ind_", "行业"), ("pro_", "职业舞者")]:
        vars_ = [i for i in res_core.index if i.startswith(prefix)]
        if not vars_:
            continue
        print(f"\n  【{label}】")
        for v in sorted(vars_, key=lambda x: res_core.loc[x, "coef"]):
            c, p = res_core.loc[v, "coef"], res_core.loc[v, "pvalue"]
            sig = "*" if p < 0.05 else ""
            name = v.replace(prefix, "")
            direction = "更靠前" if c < 0 else "更靠后"
            print(f"    {name}: β = {c:+.2f}{sig}  (p={p:.4f})  → 相对参照组排名{direction}")

    print(f"\n[完成] 系数表已保存: {out_dir / 'coef_placement_only.csv'}")


if __name__ == "__main__":
    main()
