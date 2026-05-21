# Proposal A — OM 版

## *The Operational J-Curve of Diagnostic AI: Throughput, Turnaround, and Induced Demand in Radiology*

**目标期刊（按优先级）**：M&SOM（健康运营线）→ POM → Management Science (OM dept) → JOM
**方法引擎**：FDA 器械清单（处理菜单）+ 理赔反推真实采纳 + 交错 DiD / 事件研究 / DDD / IV
**一句话论点**：AI 诊断器械的运营收益不是即时净正——它先因复核工序拖慢流程（J 曲线），诱导更多影像与下游检查（Jevons），且只在产能成熟的科室兑现。

> 说明：本文件把每一步都写成"主方案 + 替代方案"。真正执行时只挑一条主线，其余作为 robustness 或 Plan B。

---

## 1. 研究缺口与定位

**现状**：现有 AI 医疗器械的实证文献（NEJM AI 2024 等）只做了**描述性采纳曲线**，没有人把采纳接到**运营结果**上，也没有任何因果设计。OM 顶刊零篇用 FDA 清单。

**本文切口**：把放射科当作一个**服务生产系统**（影像=产品，放射科医生=server，AI=插入流程的一道新工序）。问 AI 落地后这个系统的**流量表现**怎么变。

**替代定位（若审稿人嫌"运营"不够新）**：
- (a) 把焦点收窄到**单一高价值流程**（卒中 LVO 的 door-to-treatment），做"AI 工作流加速"的窄而深版本。
- (b) 把焦点放在**容量挤兑/拥堵外溢**：AI 诱导需求是否挤占了非 AI 检查的产能。
- (c) 框成**技术采纳的运营学习曲线**论文，J 曲线为主结果。

---

## 2. 研究问题与假设

**RQ1**：AI 诊断采纳对影像服务系统的吞吐、周转、产能利用有何因果影响？
**RQ2**：效应随时间如何演化（是否存在 J 曲线）？
**RQ3**：是否诱导下游影像/手术需求（Jevons）？
**RQ4**：运营收益的边界条件是什么（按产能、人手、流程成熟度异质）？

**假设矩阵**：

| 编号 | 假设 | 机制 | 主 DV |
|---|---|---|---|
| H1 | 采纳→周转↓、吞吐↑ | 自动化省时 | 周转、吞吐 |
| H2 | 周转先升后降（J 曲线） | 复核 AI 标记的工序开销 + 学习 | 周转的事件研究动态系数 |
| H3 | 影像量、下游手术↑（Jevons） | 单次成本降→开更多 | 影像量、下游级联率 |
| H4 | 收益集中在高产能/高人手科室 | 互补产能 | 处理×基线产能交互 |
| H5 | 速度↑但质量/结局不变 | AI 只加快简单片子 | door-to-treatment vs 90 天结局 |

**替代假设（预注册时一并列出，避免 HARKing）**：
- H2'：无 J 曲线，即时单调改善（若 AI 工具高度自动化、无需人工复核）。
- H3'：无诱导需求，甚至影像量下降（若 AI 用于 triage 减少不必要检查）。
- H4'：收益集中在**低**基线产能科室（若 AI 主要补人手短板，则小科室获益更多）——与 H4 相反，数据来裁决。

---

## 3. 概念框架（理论锚 + 替代透镜）

**主透镜**：服务运营 / 排队论 —— 放射科医生是有限的 server，AI 改变到达率（诱导需求）与服务时间（复核工序），净效应取决于二者相对大小。

**替代/补充透镜**：
- 生产率 J 曲线（Brynjolfsson-Rock-Syverson）：把暂时性产能损失理论化。
- 技术-任务匹配（task-technology fit）：解释 H5（AI 只擅长某类片子）。
- 运营互补性：解释 H4（AI 需要互补产能/流程）。
- 行为运营（若做医生 override 行为）：解释为何复核工序拖慢。

---

## 4. 数据

### 4.1 处理菜单（什么 AI、何时可用）
- **主**：FDA AI/ML-Enabled Medical Devices List —— 器械名、批准日、专科、厂商、类型。
- **替代/补充**：FDA 510(k)/De Novo/PMA 数据库交叉核对批准细节；厂商上市公告校准"商业可得"时点（批准≠上市）。

### 4.2 真实采纳测量（核心，决定成败）
- **主**：CMS Medicare FFS 理赔中的 **AI 专属 CPT / NTAP 码**。
  - 门诊 CPT：FFR-CT（0501T–0504T → Cat I 75580）、糖网 AI（92229）、AI 心电（0902T）、超声心动（0932T）、胸片（0877T–0880T）等。
  - 住院 NTAP：Viz LVO 卒中（约 FY2021 起）。
  - 出现该码的首个季度 = 该医院/NPI 的采纳时点 τ。
- **替代采纳来源（互为验证或补缺）**：
  1. **AHA Annual Survey IT Supplement / 原 HIMSS Analytics**：医院层 IT/AI 采纳自报。
  2. **Definitive Healthcare / Vizient Clinical Data Base**：商业库，含设备与使用。
  3. **厂商客户名单 / 新闻稿**：粗粒度但可校准时点。
  4. **商业理赔**（Optum Clinformatics、Merative MarketScan、HealthVerity）：补 Medicare 漏掉的 65 岁以下人群。
- **NEJM AI 2024 已验证"CPT 反推采纳"可行**——引用它为测量背书，本文贡献不在测量而在"接运营结果 + 因果 + 合并 NTAP 通道"。

### 4.3 运营结果变量
- **吞吐**：单位时间影像计数（理赔可得）。
- **下游级联**：指数检查后 X 天内的后续 CPT/手术（理赔可得）→ Jevons。
- **time-to-treatment**：卒中 door-to-treatment（理赔时间序列或登记库）。
- **报告周转时间（turnaround）**：理赔**没有** → 替代方案：
  1. 疾病登记库（GWTG-Stroke 的 door-to-needle/door-to-puncture）。
  2. 单一医院系统/合作方提供的 RIS/PACS 时间戳（深但样本窄）。
  3. 用理赔时间间隔（影像→下一步服务的间隔）做**代理**。
  4. 退而求其次：放弃周转，只用吞吐+级联+time-to-treatment（仍足以支撑 H2 的"动态"用 door-to-treatment 体现）。

### 4.4 协变量
- HCRIS 成本报告（床位、FTE、教学、产权、case mix）。
- POS 文件、AHA 调查（结构特征）。
- Area Health Resource File / ADI（社区 SES，供公平异质性）。

### 4.5 数据获取路径（替代）
- **主**：CMS VRDC（虚拟研究数据中心，远程访问全样本）。
- **替代**：ResDAC 申请提取 Medicare 文件（Carrier、Outpatient、MedPAR）；或先用 Medicare 5% 样本探路降成本；或商业理赔（采购但无需 IRB/DUA 等待）。

---

## 5. 面板构建（替代单位/时间）

- **主**：医院(CCN) × 季度，2016–2025；强烈建议堆叠成 **医院 × modality × 季度**（DDD 要用）。
- **替代单位**：放射科医生(NPI) × 季度（更细，但采纳定义更噪）；医院系统层（更粗，控并购）。
- **替代时间粒度**：月度（功效高但季节噪大）；半年（噪小但动态分辨率低）。
- **采纳时点 τ 定义的替代**：首次出现码 / 累计达阈值（如 ≥10 次，过滤试用噪声）/ 首次连续两季出现（稳态采纳）。

---

## 6. 识别策略（主 + 全部替代 + 稳健）

**核心威胁**：选择性采纳——采纳医院本就更大、更数字化。朴素 TWFE 会被异质处理效应污染。

**主识别**：交错 DiD，用 **not-yet-treated** 作对照。
- 估计量替代（互为 robustness）：Callaway & Sant'Anna (2021)、Sun & Abraham (2021)、de Chaisemartin & D'Haultfœuille (2020)、Borusyak-Jaravel-Spiess 插补、Gardner 两阶段、Cengiz 等堆叠 DiD。
- 用 Goodman-Bacon (2021) 分解暴露坏比较的占比。

**平行趋势检验与敏感性**：
- 事件研究 leads 检验 pre-trend。
- Roth (2022)：报告 pre-trend 检验的功效。
- Rambachan & Roth (2023) honest DiD：对平行趋势违背做敏感性区间。

**最强一招——三重差分 (DDD)**：医院内"AI 相关 modality vs 非 AI modality"作内部对照，差掉医院级同期冲击（IT 升级、并购、医保政策）。

**应对"何时采纳"内生性的 IV（替代/补充）**：
- 主：NTAP 生效时点（全国固定日期）× 医院基线暴露（卒中量/Medicare 占比）作 shift-share/Bartik 式工具。
- 替代：相邻医院采纳率（同行扩散，慎防违反排他性）；厂商分区上市顺序。
- **你的 [9] Lewbel (2012) 异方差识别**：外部工具弱时，用采纳方程异方差构造内生工具——方法学差异化卖点，放稳健性。

**其他设计替代**：
- 合成控制 / 合成 DiD（Arkhangelsky 等 2021）：处理组少时用。
- 匹配 + DiD：CEM / 熵平衡（Hainmueller）/ PSM 在基线运营+结构协变量上匹配后再 DiD。
- 回归不连续（若某采纳门槛存在，如 NTAP 资格线）。

---

## 7. 估计方程

事件研究主回归：

```
Y_{h,t} = α_h + λ_t + Σ_{k≠-1} β_k · 1{t − τ_h = k} + X_{h,t} γ + ε_{h,t}
```
- α_h 医院 FE，λ_t 时间 FE；k<0 检验 pre-trend，k≥0 是动态 ATT；用 C-S 聚合（not-yet-treated 对照）。

**DDD**：加 modality 维度 m，三重交互 `Treat_h × Post_t × AImodality_m`。

**IV/2SLS**：`Adopt_{h,t}` 用 `NTAP_t × BaselineExposure_h` 工具化。

**推断（替代）**：医院层聚类标准误；处理簇少→wild cluster bootstrap（Cameron-Gelbach-Miller）；地理外溢→Conley 空间 SE；随机化推断（permutation）作补充。

---

## 8. 异质性与机制

- **H4 运营能力**：处理 × 基线产量 / 人手余量 / 教学 / IT 能力交互；或 C-S 分组 ATT。
- **机制分解**：周转上升是否由"复核工序"驱动——用每例阅片时间、override 率（若有 RIS 数据）。
- **H3 机制**：诱导需求是来自医生行为（更多开单）还是患者侧——按转诊来源分。
- **替代**：用因果森林 / 异质处理效应机器学习（Wager-Athey GRF）做数据驱动的子组发现，再人为验证。

---

## 9. 证伪 / 安慰剂 / 稳健（穷举）

- 安慰剂结果：AI 不该影响的无关科室运营指标。
- 安慰剂时点：随机假采纳日。
- DDD 内部检验：非 AI modality 应当无效应（若同步动→是医院级冲击，本文垮，见 §11）。
- 留一法（leave-one-modality-out / leave-one-state-out）。
- 不同 τ 定义、不同窗口、不同估计量交叉验证。
- 选择性进出（医院退出/关闭）做 attrition 检验。

---

## 10. 预期发现与结果矩阵（好/坏/混合都各有 contribution）

| 情形 | 结果 | contribution 句 | 去向 |
|---|---|---|---|
| 最可能 | H2+H3+H4 成立 | "AI 运营收益非即时、诱导净需求、仅大医院兑现" | 主线，M&SOM/POM |
| AI 部分好 | 速度↑但质量不变（H5） | "AI 加速但不增质——速度≠质量边界" | 仍是 full paper |
| 异质主导 | 大医院↑、小医院↓（H4） | "AI 扩大运营不平等" | full paper + 公平角度 |
| 干净正结果 | 即时单调改善、量未涨 | "首个 AI 影像运营收益的因果证据（同质即时）" | letter / short report 仍可发 |

**关键认识**：贡献 = 干净识别出 effect 的**形状**（哪儿/何时/哪个维度/对谁），与平均效应的正负无关。多维 DV + 异质性几乎保证有故事。

---

## 11. 什么会真正杀死本文（A 类致命 vs B 类无聊）

- **A 类（致命，与结果好坏无关）**：① pre-trend 不平 → 因果声明垮（缓解：honest DiD 敏感性、换 not-yet-treated 对照、IV）；② DDD 非 AI modality 同步动 → 归因不到 AI（缓解：换内部对照、加 IV、合成控制）。
- **B 类（不致命）**："结果不够反直觉"——已被多维+异质性+"首个因果估计"三重保险化解，不构成威胁。

---

## 12. 功效与可行性

- **约束**：AI 报销码多 2020 后才有，早年用量薄、集中在 FFR-CT 与糖网两类。
- **缓解**：把窗口拉到 2024–2025；先在用量最大的 FFR-CT/糖网跑通，再扩卒中 NTAP；必要时合并商业理赔扩样本。
- **功效模拟**：用历史采纳速度做 ex-ante power calculation，决定最小可检测效应。

---

## 13. 局限与边界

- Medicare-only → 外部效度（缓解：商业理赔补稳健）。
- 采纳码可能滞后于真实部署（缓解：AHA IT 调查交叉验证）。
- turnaround 数据稀缺（见 §4.3 替代）。

---

## 14. 协作与技能

- 需要 CMS 数据使用协议（DUA）/ IRB；建议拉一位有 VRDC 经验的健康服务研究合作者。
- 计量上你的 [9] 是差异化武器。
- 临床合作者校准 modality–CPT 映射与临床合理性。

---

## 15. 关键参考（按主题）

- 交错 DiD：Callaway & Sant'Anna (2021); Sun & Abraham (2021); de Chaisemartin & D'Haultfœuille (2020); Goodman-Bacon (2021); Borusyak-Jaravel-Spiess; Cengiz et al. (2019).
- 平行趋势：Roth (2022); Rambachan & Roth (2023).
- 异方差识别：Lewbel (2012)。
- 合成控制：Arkhangelsky et al. (2021)。
- 采纳测量先例：NEJM AI (2024) Characterizing the Clinical Adoption of Medical AI through U.S. Insurance Claims.
- 健康运营实证范式：相关 M&SOM/POM 健康运营文献。
