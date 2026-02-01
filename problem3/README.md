# 问题三：影响因素分析

对应题目第三问：

> 利用包含估算观众投票数的数据，构建模型分析专业舞者及名人选手特征（年龄、行业等）对比赛结果的影响。这些因素对名人选手的比赛表现影响程度如何？**它们对评委打分和观众投票的影响是否一致？**

本目录提供两个脚本，分别回答两个子问题：

| 脚本 | 回答的问题 | 数据层级 | 因变量 |
|------|------------|----------|--------|
| **placement_model.py** | 这些因素对**最终排名**的影响有多大？ | 人-季 (celebrity, season) | placement |
| **impact_analysis_ranked.py** | 这些因素对**评委排名**与**观众排名**的影响是否一致？ | 人-季-周 (celebrity, season, week) | judge_rank_norm / audience_rank_norm |

两个脚本使用的**自变量 X 同套**：年龄组、国家、行业、职业舞者（与题目“同级别因素”一致，不做嵌套比较）。

---

## 1. placement_model.py（第一小问：最终排名）

### 目标

分析职业舞者 + 名人静态特征（年龄、行业、国家等）对**最终表现（最终排名 placement）**的影响有多大。不依赖淘汰模型或粉丝份额，仅用原始数据。

### 数据

- **样本单位**：每人每季一行 (celebrity, season)
- **因变量**：`placement`（最终排名，数值越小越好）
- **自变量**：赛季（固定效应）+ 年龄组 + 国家 + 行业 + 职业舞者

### 模型

- **一个 OLS**：`placement ~ 赛季(FE) + 年龄组 + 国家 + 行业 + 职业舞者`
- 年龄组参照：&lt;30 岁；国家/行业：Top 5 / Top 8 + Other；职业舞者：跨赛季出现 ≥2 次单独编码，其余 Other_Partner

### 输出

| 文件 | 说明 |
|------|------|
| `coef_placement_only.csv` | 各因素系数与 p 值（不含赛季 FE）；**负系数 = 排名更靠前** |
| `placement_effects.png` | 四格图：年龄组、国家、行业、职业舞者系数条形图 |

### 稳健性

- 另跑一版：因变量改为**赛季内 min-max 归一化排名**，报告 Adjusted R²，验证结论对尺度不敏感。

---

## 2. impact_analysis_ranked.py（第二小问：评委 vs 观众）

### 目标

分析同一套因素（年龄组、国家、行业、职业舞者）对**当周评委排名**与**当周观众排名**的影响是否一致。比较两个模型的系数方向与显著性。

### 数据

- **样本单位**：每人每季每周一行 (celebrity, season, week)，不聚合
- **因变量（周内归一化排名）**：`rank_norm = (rank - 1) / (n - 1)`，0=当周最好、1=当周最差，跨周可比
  - **y₁ = judge_rank_norm**：由当周评委总分周内排序得到 judge_rank，再归一化（原始数据）
  - **y₂ = audience_rank_norm**：由估计的观众排名 audience_rank 得到，再归一化（**观众端为估计值，非官方真实票数**）
- **自变量**：与 placement_model 同套（年龄组、国家、行业、职业舞者）+ **season-week 固定效应**
- **标准误**：按 (celebrity, season) 聚类的稳健标准误

### 观众解耦（按赛季类型）

观众排名/份额按题目规则分赛季类型构造：

| 赛季 | 方法 | 说明 |
|------|------|------|
| **3–27** | 份额法（和=2） | estimated_combined_share = 2×survival_prob/Σ；audience_share_raw = 综合份额 − judge_share；周内平移+归一化 → audience_share；audience_rank = 按 audience_share 周内排名 |
| **1–2，28+** | 排名法 n(n+1) | combined_rank_score = norm(elim_prob)×n(n+1)；audience_rank_score = combined_rank_score − judge_rank；audience_rank = 按 audience_rank_score 周内排名；audience_share 由排名反推并周内归一化（有并列时和=1） |

淘汰概率 `elimination_prob` 来自同一淘汰模型（FEATURE_COLS_ELIM + is_eliminated），与 main 流程一致。

### 模型

- **模型 1**：`judge_rank_norm ~ X + season-week FE`，cluster (celebrity, season)
- **模型 2**：`audience_rank_norm ~ X + season-week FE`，cluster (celebrity, season)
- **交互检验**：堆叠两条 y，加 `type_audience` 与 X×type_audience；交互项显著表示该因素在评委/观众端影响方式不同

### 输出

| 文件 | 说明 |
|------|------|
| `coef_judge_rank_norm.csv` | 评委模型系数（仅年龄/国家/行业/职业舞者）；**正系数 = 当周排名更靠后** |
| `coef_audience_rank_norm.csv` | 观众模型系数（同上）；观众排名为估计值 |
| `judge_vs_audience_effects.png` | 评委 vs 观众系数对比图（纵向条形，英文，* p&lt;0.05） |
| `interaction_judge_vs_audience.csv` | X×type_audience 交互项系数与 p 值 |

### 解读

- 比较两表：**同向且都显著** → 评委与观众对该因素反应一致；**仅一端显著或符号相反** → 不一致
- 交互表：**p&lt;0.05 的交互项** → 该因素在评委端与观众端的影响方式存在统计差异

---

## 3. 自变量 X（两脚本共用逻辑）

| 类型 | 构造 | 参照/说明 |
|------|------|-----------|
| 年龄组 | 30/40/50 切分 | 参照 &lt;30 |
| 国家 | Top 5 + Other_Country | 按人-季一条计 Top5（impact 脚本）；placement 按人-季计 |
| 行业 | Top 8 + Other | 同上 |
| 职业舞者 | 跨赛季出现 ≥2 次单独编码，其余 Other_Partner | 与 placement 一致 |

---

## 4. 依赖与运行

### 依赖

```
pip install numpy pandas statsmodels
```

可选：`matplotlib`（出图）、`scipy`（无 statsmodels 时 p 值）。

### 运行

在项目根目录下：

```bash
# 第一小问：最终排名
python problem3/placement_model.py

# 第二小问：评委 vs 观众
python problem3/impact_analysis_ranked.py
```

---

## 5. 重要说明（观众端为估计）

- **judge_rank**：由原始每周评委分数在当周内排序得到，为观测数据。
- **audience_rank / audience_share**：由淘汰概率模型反推得到（份额法或排名法），为**估计值**，非官方真实票数。报告与结论中需明确写出“观众端为估计”。

---

## 6. 输出文件汇总

| 脚本 | 输出文件 |
|------|----------|
| placement_model.py | `coef_placement_only.csv`, `placement_effects.png` |
| impact_analysis_ranked.py | `coef_judge_rank_norm.csv`, `coef_audience_rank_norm.csv`, `judge_vs_audience_effects.png`, `interaction_judge_vs_audience.csv` |
