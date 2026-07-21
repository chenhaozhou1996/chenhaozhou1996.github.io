# Negative-Control Methods for Panel Fixed-Effects Designs: A Fact-Checked Literature Audit

**Purpose.** This is a source-verified literature audit supporting a methods paper that proposes a four-phase *Specify & Nominate → Detect → Correct → Triangulate & Report* workflow for using negative control outcomes (NCO) and negative control exposures (NCE) to detect and correct unobserved confounding in panel-data fixed-effects (FE) designs, aimed at empirical operations management (OM) and healthcare operations.

**What this document does.** (1) Verifies every citation in the workflow figure against primary sources — exact authors, venue, year, pages, DOI — and flags the ones that need correction or re-attribution; (2) maps the 2010–2026 state of the art onto the four phases; (3) catalogs the failure modes and critiques the paper must confront; (4) positions the paper against competing/complementary frameworks and assesses the central gap claim; (5) gives a concrete revision checklist.

**Method & confidence.** Findings rest on a fan-out web-search + adversarial-verification pass (23 primary sources fetched, 105 candidate claims extracted, 25 verified by 3-vote adversarial review — 24 confirmed unanimously, 1 refuted), supplemented by targeted verification of the two OM-side citations the automated pass did not reach (Ketokivi & McIntosh 2017; the "Lu et al. 2018" identity). Verification sources are primary journal pages, PubMed/PMC records, arXiv, JMLR, and the method originators' Duke-Margolis workshop deck. Two caveats are carried throughout: (a) the Duke deck, used for cross-checking attributions, was authored by the proximal-inference originators and is therefore authoritative but not independent; (b) the paper's central *gap* claim is consistent with — but not positively proven by — the evidence, because all correction-side literature verified here sits in epidemiology/biostatistics/ML venues.

---

## 1. Bottom line

**Every one of the 13 citations in the workflow figure is real and, with three attribution nuances noted below, correctly used.** The workflow rests on a coherent, defensible 2010–2026 arc:

- **Detection** is founded on Lipsitch, Tchetgen Tchetgen & Cohen (2010), which is the paper that *defines* NCE vs. NCO, U-comparability, and the exact falsification rule Phase II uses.
- **Single-NCO calibration/correction** traces to Tchetgen Tchetgen's Control Outcome Calibration Approach (2014), generalized to difference-in-differences by Sofer et al. (2016).
- **NCE+NCO proximal correction** rests on Miao, Geng & Tchetgen Tchetgen (2018), with Liu et al. (2025) supplying the practical regression/2SLS implementation.
- **Data-driven discovery** is Kummerfeld, Lim & Shi's DANCE (2024).
- **Triangulation** cites correctly attributed sensitivity tools (Oster 2019; Cinelli & Hazlett 2020; Busenbark et al. 2022) plus the OM-facing endogeneity review by Lu et al. (2018).

**Four things to change or add before the paper goes out:**

1. **Re-attribute "additive equi-confounding."** The figure reads "calibration under additive equi-confounding *per Tchetgen Tchetgen 2014*." Additive equi-confounding is precisely defined and named in **Sofer et al. (2016)**, not Tchetgen Tchetgen (2014), whose COCA rests on a *rank-preservation*-type assumption. Cite Tchetgen Tchetgen (2014) for COCA/calibration generally; attribute additive equi-confounding (and its parallel-trends equivalence) to Sofer et al. (2016).
2. **Confirm the identity of "Lu et al. 2018."** It is almost certainly **Lu, Ding, Peng & Chuang (2018), *Journal of Operations Management* 64(1):53–64** — the OM endogeneity review — *not* Lu & White (2014) as an automated match suggested. (Co-author Xin Ding is on the author's committee; this is the natural referent.) Verify the exact intended source and fix if needed.
3. **Add the equivalence-testing lineage to Phase II.** The "never a bare p > 0.10 — report an equivalence bound" prescription has a direct methodological ancestor the figure omits: **Hartman & Hidalgo (2018)** (equivalence approach to balance/placebo tests) and, for the panel/DiD power argument, **Roth (2022)** (pre-test power and pre-testing bias). These make the equivalence-bound rule a cited standard rather than an assertion.
4. **Engage the nearest prior work on panel proximal inference.** **Ying, Miao, Shi & Tchetgen Tchetgen (2023, JRSS-B)** already extends proximal causal inference to *complex longitudinal* (FE-adjacent) settings. Because the paper's novelty is the panel-FE framing, it must cite and distinguish this work explicitly — it is the closest existing method to the paper's territory and bears directly on the gap claim.

**Two interpretive cautions:** (i) **Danieli et al. (2026)** is an *instrumental-variable* falsification paper and a *caution* about false positives, not a panel-FE endorsement of NCO detection — cite it as motivating equivalence bounds, not as validating naive falsification. (ii) Do **not** attribute the "single proxy → detect / two proxies → correct" division to Miao et al. (2018); that specific mapping was adversarially refuted. Anchor detection to Lipsitch (2010) and correction to Miao (2018) + Tchetgen Tchetgen (2014).

---

## 2. Citation verification table

All 13 figure citations, verified. "Status" is ✓ (verified as used), ⚠ (verified but attribution/label needs a fix), or ↔ (identity re-assigned from an automated mis-match).

| # | Figure citation | Verified reference | Venue / year / locator | Status |
|---|---|---|---|---|
| 1 | Ketokivi & McIntosh 2017 (Fig. 2a) | Ketokivi, M., & McIntosh, C. N. "Addressing the endogeneity dilemma in operations management research: Theoretical, empirical, and pragmatic considerations." | *J. Operations Management* 52:1–14, 2017. DOI 10.1016/j.jom.2017.05.001 | ✓ |
| 2 | Lipsitch et al. 2010 | Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. "Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies." | *Epidemiology* 21(3):383–388, 2010. PMID 20335814 | ✓ |
| 3 | Eggers et al. 2024 | Eggers, A. C., Tuñón, G., & Dafoe, A. "Placebo Tests for Causal Inference." | *Am. J. Political Science* 68(3):1106–1121, 2024. DOI 10.1111/ajps.12818 | ✓ |
| 4 | Danieli et al. 2026 | Danieli, O., Nevo, A., Walk, I., Weinstein, B., & Zeltzer, D. "Negative Control Falsification Tests for Instrumental Variable Designs." | *American Economic Review* 116(4):1380–1414, 2026. DOI 10.1257/aer.20240636 (preprint arXiv:2312.15624) | ✓ (read as a caution) |
| 5 | Tchetgen Tchetgen 2014 | Tchetgen Tchetgen, E. J. "The Control Outcome Calibration Approach for Causal Inference With Unobserved Confounding." | *Am. J. Epidemiology* 179(5):633–640, 2014. DOI 10.1093/aje/kwt303 | ⚠ (see equi-confounding note) |
| 6 | Sofer et al. 2016 | Sofer, T., Richardson, D. B., Colicino, E., Schwartz, J., & Tchetgen Tchetgen, E. J. "On Negative Outcome Control of Unobserved Confounding as a Generalization of Difference-in-Differences." | *Statistical Science* 31(3):348–361, 2016. DOI 10.1214/16-STS558; PMID 28239233 | ✓ (owns additive equi-confounding) |
| 7 | Miao et al. 2018 | Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. "Identifying causal effects with proxy variables of an unmeasured confounder." | *Biometrika* 105(4):987–993, 2018. DOI 10.1093/biomet/asy038 | ✓ |
| 8 | Liu et al. 2025 | Liu, J., Park, S., Li, K., & Tchetgen Tchetgen, E. J. "Regression-Based Proximal Causal Inference." | *Am. J. Epidemiology* 194(7):2030–2036, 2025. DOI 10.1093/aje/kwae370 (online Sep 2024; arXiv:2402.00335) | ⚠ (GLM generalization, not the 2SLS origin) |
| 9 | Kummerfeld et al. 2024 (DANCE) | Kummerfeld, E., Lim, S., & Shi, X. "Data-driven Automated Negative Control Estimation (DANCE)." | *J. Machine Learning Research* 25:1–35, 2024 (arXiv:2210.00528) | ✓ |
| 10 | Oster 2019 | Oster, E. "Unobservable Selection and Coefficient Stability: Theory and Evidence." | *J. Business & Economic Statistics* 37(2):187–204, 2019. DOI 10.1080/07350015.2016.1227711 | ✓ |
| 11 | Busenbark et al. 2022 | Busenbark, J. R., Yoon, H., Gamache, D. L., & Withers, M. C. "Omitted Variable Bias: Examining Management Research With the Impact Threshold of a Confounding Variable (ITCV)." | *J. Management* 48(1):17–48, 2022. DOI 10.1177/01492063211006458 | ✓ |
| 12 | Cinelli & Hazlett 2020 | Cinelli, C., & Hazlett, C. "Making Sense of Sensitivity: Extending Omitted Variable Bias." | *J. Royal Statistical Society: Series B* 82(1):39–67, 2020. DOI 10.1111/rssb.12348 | ✓ |
| 13 | Lu et al. 2018 | **Lu, G., Ding, X. (David), Peng, D. X., & Chuang, H. H.-C.** "Addressing endogeneity in operations management research: Recent developments, common problems, and directions for future research." | *J. Operations Management* 64(1):53–64, 2018. DOI 10.1016/j.jom.2018.10.001 | ↔ (re-identified; not Lu & White 2014) |

### Per-citation notes (the ones that matter)

**#2 Lipsitch et al. 2010 — the load-bearing detection citation, verified verbatim.** Abstract: "We distinguish 2 types of negative controls (exposure controls and outcome controls) … identify the conditions for the use of such negative controls to detect confounding." The detection rule the paper's Phase II implements is stated word-for-word: "If N and Y are U-comparable outcomes (ie, with an identical set of common causes that are associated with A), and assuming that N is not caused by A, an association A–N when analyzed according to the same procedure used to analyze A–Y would indicate bias in the association A–Y." Crucially — and this is the paper's justification for equivalence bounds over a bare non-significant p — the test is **asymmetric and fallible in both directions**: real NCOs are "only approximately U-comparable, at best," so a detected A–N association can be driven by a *distinct* confounder and does not prove A–Y is biased, and a null A–N association does not prove A–Y is unbiased if N shares only some of A–Y's confounders. The three threats the paper argues against in Phase I nomination (coupling, anticipation, feedback) are operationalizations of this paper's U-comparability + no-direct-effect requirements.

**#5/#6 The equi-confounding attribution needs to move.** Tchetgen Tchetgen (2014) proposes COCA as "a simple but formal counterfactual or potential outcome-based approach to correct causal effect estimates for bias due to unobserved confounding," and supplies the field-standard **valid-NCO definition** the paper should use in Phase I: an outcome is a valid negative control "to the extent that it is influenced by unobserved confounders of the exposure effects on the outcome in view, although not directly influenced by the exposure." But COCA's identifying assumption in its linear/constant-effect form is **rank preservation** (the authors call it "a strong assumption"), *not* additive equi-confounding. Additive equi-confounding — formally `E{Y⁰|A=1,C} − E{Y⁰|A=0,C} = E{N|A=1,C} − E{N|A=0,C}` — is defined and named in **Sofer et al. (2016)**, which proves that difference-in-differences *is* a negative-outcome-control design ("the pre-exposure outcome is a negative control outcome, as it cannot be influenced by the subsequent exposure, and it is affected by both observed and unobserved confounders") and that this "provides simple conditions under which negative control outcomes can be used to detect *and* correct for confounding bias." Sofer et al. themselves qualify additive equi-confounding as "overly restrictive and rarely credible, because it requires that both the outcome of interest and the control outcome are measured on the same scale" — which is exactly the honesty the paper's Task-3 critique wants. **Action:** figure caption should read "calibration under additive equi-confounding (Sofer et al. 2016; COCA more generally, Tchetgen Tchetgen 2014)."

**#8 Liu et al. 2025 is the *regression/GLM generalization*, not the 2SLS origin.** Confirmed identity: Jiewen Liu, Sungho Park, Kendrick Li, Eric Tchetgen Tchetgen, "Regression-Based Proximal Causal Inference," *AJE* 194(7):2030–2036. It "de-bias[es] confounded causal effect estimates by leveraging a pair of treatment and outcome negative control or confounding proxy variables" via "2-stage generalized linear regression models (GLMs) … applicable to continuous, count, and binary outcomes," implemented in the `pci2s` (proximal causal inference 2SLS) package. If Phase III's claim is specifically about *proximal two-stage least squares*, cite Liu et al. (2025) **alongside** the 2SLS origin in the proximal-causal-learning framework (Tchetgen Tchetgen, Ying, Cui, Shi & Miao 2020; Cui et al. 2024, *Statistical Science*), not in place of it.

**#4 Danieli et al. 2026 — cite it as a critique, not an endorsement.** It is the flagship *economics* adoption of negative controls (top journal, April 2026), which is genuinely useful evidence that NCO/NCE methods have crossed into economics. But two nuances matter for how the paper cites it: (a) it concerns **instrumental-variable designs**, not panel fixed effects; (b) its message is a **caution** — negative-control falsification tests are conditional-independence tests that *also* probe functional-form assumptions, so "conventional applications may flag problems even in valid designs" (false positives), and common tests require conditioning on the instrument (often skipped). This *supports* the "never a bare p > 0.10 / report equivalence bounds" stance, but as a warning about naive falsification, not a straightforward validation of NCO detection.

**#13 "Lu et al. 2018" is the JOM endogeneity review.** An automated citation-match flagged no clean "Lu et al. 2018" and offered Lu & White (2014, *J. Econometrics*, robustness tests) as a thematic stand-in. That is very likely wrong. The natural referent — multi-author ("et al."), 2018, operations-management audience, endogeneity-focused, sitting comfortably beside Oster/Busenbark/Cinelli–Hazlett as the "here is how OM handles endogeneity" umbrella in Phase IV — is **Lu, Ding, Peng & Chuang (2018), JOM 64(1):53–64**. Co-author Xin Ding's role on the author's committee makes this all but certain. Confirm against the paper's own bibliography and correct any "Lu & White" placeholder.

---

## 3. State of the art, 2010–2026, mapped to the four phases

The literature independently organizes itself into the same detect/calibrate/correct structure the paper uses. Two reviews establish this: **Shi, Miao & Tchetgen Tchetgen (2020, *Current Epidemiology Reports*)** — "a selective review" spanning "detection, reduction, and correction of confounding bias" plus proximal identification — and a **scoping review (Yang et al., *J. Clinical Epidemiology*, 2023/24, PMID 38040387)** that taxonomizes 37 methodological articles into three functions: **bias detection (37.8%, n=14), P-value/CI calibration (13.5%, n=5), and bias correction (43.3%, n=16)**, noting that "bias correction has been the most studied methodologically."

### Phase I — Specify & Nominate
- **Confounder inventory / theory-first specification:** Ketokivi & McIntosh (2017) is the correct OM anchor — its thesis that "endogeneity is not a problem that can be solved" motivates a design that *bounds and triangulates* rather than claims to eliminate confounding.
- **Valid-NCO definition for nomination:** Tchetgen Tchetgen (2014) supplies the exclusion (no direct effect) + U-comparability criteria; Lipsitch et al. (2010) supplies the exposure-vs-outcome distinction and the "same set of common causes" requirement that the coupling/anticipation/feedback threats operationalize.

### Phase II — Detect (falsification with teeth)
- **Rule:** Lipsitch et al. (2010) — run the NCO through the identical specification; any exposure–NCO association signals residual confounding.
- **Why equivalence bounds, not a bare p:** the direct ancestor is **Hartman & Hidalgo (2018, *AJPS* 62(4):1000–1013)** — traditional placebo/balance tests use the *wrong null* (no difference), so nonsignificance is misread as validity; they reverse the null (start from "design is invalid," require positive equivalence-bound evidence), with the `equivtest` package. **This is the citation the figure's "never a bare p > 0.10" most needs and currently lacks.**
- **Power/pre-testing critique in the panel-FE/DiD setting the paper targets:** **Roth (2022, *AER: Insights* 4(3):305–322)** — pre-trend tests catch consequential violations only ~50% of the time, and conditioning on "passing" distorts point estimates and CI coverage. Eggers, Tuñón & Dafoe (2024) provide the general placebo-test typology and the caution that the extra assumptions needed to make placebo tests informative "can be strong." Danieli et al. (2026) add the false-positive/functional-form caution for the IV case. The epistemic ceiling common to all of these is stated cleanly by *Falsification before Extrapolation* (Hussain et al., NeurIPS 2022): a falsification test "can give decisive evidence when an assumption fails but cannot prove it holds" — passing is non-refutation, not validation.

### Phase III — Correct
- **One valid NCO → calibration:** Tchetgen Tchetgen (2014) COCA; the DiD special case and additive-equi-confounding identification, Sofer et al. (2016).
- **NCE+NCO pair → proximal identification / 2SLS:** Miao, Geng & Tchetgen Tchetgen (2018) prove nonparametric identification "with at least two independent proxy variables satisfying a certain rank condition … even if the measurement error mechanism … may not be identified." The formal framework is *proximal causal learning* (Tchetgen Tchetgen, Ying, Cui, Shi & Miao 2020, arXiv:2009.10982; expanded in *Statistical Science* 2024), which treats measured covariates as "imperfect proxies of confounding mechanisms." The practical regression/GLM implementation is Liu et al. (2025).
- **Panel/longitudinal extension (nearest prior work):** **Ying, Miao, Shi & Tchetgen Tchetgen (2023, *JRSS-B* 85(3):684)**, "Proximal causal inference for complex longitudinal studies," develops proximal recursive 2SLS for point and time-varying treatments, consistent under a linear outcome-model restriction. This is the closest existing method to the paper's panel-FE positioning and must be engaged directly.
- **Many candidates → data-driven discovery:** Kummerfeld, Lim & Shi (2024) DANCE — a statistical test to discover "disconnected negative controls" that surrogate the unmeasured confounder, folded into an ATE-aggregation algorithm, claiming (hedged) to be "the first data-driven method to validate negative controls."

### Phase IV — Triangulate & Report
- **Coefficient stability (econ):** Oster (2019) — joint movement of coefficient and R² across specifications (parameters δ and R²max) yields a bias-adjusted estimate/bound.
- **Robustness values (stats):** Cinelli & Hazlett (2020) — the minimum partial-R² a confounder needs with *both* treatment and outcome to overturn a result, computable from standard output; `sensemakr`, with an IV extension.
- **ITCV (management):** Busenbark et al. (2022) — Frank's Impact Threshold of a Confounding Variable, operationalized for the paper's target discipline.
- **OM endogeneity umbrella:** Lu, Ding, Peng & Chuang (2018).
- **FE-native and cross-disciplinary complements the figure should consider adding:** honest DiD partial-identification bounds under bounded parallel-trends violations (**Rambachan & Roth 2023, *REStud* 90(5):2555–2591**, `HonestDiD`) — arguably *the* sensitivity benchmark for the DiD/panel-FE setting; and the **E-value (VanderWeele & Ding 2017, *Ann. Intern. Med.*)** — the risk-ratio-scale analogue of Oster's δ / the robustness value / ITCV, useful for bridging the healthcare-ops audience.

### Adoption: economics/management vs. epidemiology
The detect+correct framing is **mature in epidemiology/biostatistics** (the reviews above) and has **recently crossed into economics** (Danieli et al. 2026, for IV). On the **OM/management** side, the identification canon does *not* yet include negative controls or proximal inference: the MSOM OM Forum on causal inference in OM presents IV, DiD, RD, matching, and event studies with **no** negative-control or proximal component, and the JOM endogeneity reviews (Ketokivi & McIntosh 2017; Lu et al. 2018) predate the proximal literature. What management is currently reaching for on the unobserved-confounding-robustness frontier is **partial identification/bounds** (Frake et al. 2025, *SMJ*, "From Perfect to Practical"), not NCO/NCE correction. Proximal methods remain non-standard enough outside biostatistics that 2025 saw practitioner "demystifying" guides (Ringlein et al. 2025).

---

## 4. Failure modes and critiques (Task 3)

The verified sources establish a **monotone assumption-ladenness gradient** across the phases — every correction method rests on at least one *untestable* identifying assumption, which is the single strongest justification for the paper's Phase IV triangulation.

1. **Detect (Lipsitch 2010)** — lightest. Rests on *approximate* U-comparability, which is untestable, and yields an asymmetric test that is fallible in **both** directions (false alarms from a distinct confounder; false reassurance when the NCO shares only some confounders). This is why the paper is right to demand an equivalence bound and a magnitude/CI, not a bare non-significant p.
2. **Single-NCO calibration (Tchetgen Tchetgen 2014 / Sofer 2016)** — adds **additive equi-confounding**, self-described as "overly restrictive and rarely credible," requiring the outcome and control outcome on the *same scale* (COCA's linear form additionally embeds rank preservation). The scoping review adds that correction algorithms generally need "rank preservation, monotonicity, and linearity."
3. **Proximal NCE+NCO (Miao 2018 / Liu 2025)** — adds an **untestable completeness/rank condition** plus a **relevance requirement**: the proxies must be "sufficiently informative about confounding." This is a **weak-proxy problem directly analogous to weak instruments** — weak proxies yield inconsistent estimates and invalid inference with inflated variance in proximal 2SLS, and existing PCI methods require *all* proxies to be valid and can be severely biased if even one violates the exclusion condition (see the 2025 "invalid/weak proxy" cluster: *Adaptive Proximal Causal Inference with Some Invalid Proxies*, arXiv:2507.19623; *Fortified PCI with Many Invalid Proxies*, arXiv:2506.13152). Phase III must therefore report a **relevance/first-stage-strength diagnostic** and an invalid-proxy robustness check.
4. **Data-driven discovery (DANCE 2024)** — heaviest structural burden. Validity is proven **only** for the restrictive "disconnected negative control" class under a **linear, acyclic, continuous "simple NC model"** plus an **untestable Tetrad Faithfulness** assumption; DANCE "will throw out some NCs that satisfy a DAG different than [its Figure 1]" (i.e., it *discards valid controls of other structures*) and "relies on at least three disconnected NCs." It is **not panel-FE-specific**. Its validation test uses vanishing-tetrad / rank-deficiency tests.

**Open failure mode not covered by any verified source — the paper should address it itself: multiple testing.** Nominating and testing many candidate NCOs (Phase I → Phase II) is a multiple-comparisons problem the negative-control literature verified here does not treat. The paper should specify how it handles this — e.g., pre-registering the candidate set and its exclusion/U-comparability arguments before looking at exposure–NCO associations, applying a family-wise or FDR correction to the falsification tests, or adopting DANCE's built-in validation-test framing when ≥3 candidates exist. Presenting a defensible answer here is a genuine contribution, because the field has largely left it open.

**One refuted claim to avoid.** The proposition that Miao et al. (2018) justifies mapping "single proxy → detect / two proxies → correct" onto the Phase II/Phase III split was **adversarially refuted 0–3**. Miao (2018) is an *identification* result about two proxies; it does not license attributing the detect-vs-correct division to itself. Keep detection anchored to Lipsitch (2010) and correction to Miao (2018) + Tchetgen Tchetgen (2014)/Sofer (2016).

---

## 5. Competing / complementary frameworks and the gap claim (Task 4)

**Complementary sensitivity frameworks** (these *bound* rather than *point-correct*, so they belong in Phase IV alongside — not instead of — negative-control correction):

| Framework | Reference | Discipline home | What it delivers |
|---|---|---|---|
| Coefficient stability (δ, R²max) | Oster 2019 | Economics | Bias-adjusted point/bound from coefficient + R² movement |
| Robustness value (partial R²) | Cinelli & Hazlett 2020 | Statistics | Min. confounder strength (with treatment *and* outcome) to overturn; `sensemakr` |
| ITCV | Busenbark et al. 2022 | Management | Min. confounder correlations to invalidate an inference |
| E-value | VanderWeele & Ding 2017 | Epidemiology | Risk-ratio-scale min. association to explain away an estimate |
| Honest DiD bounds | Rambachan & Roth 2023 | Econometrics | Partial ID under bounded parallel-trends violation; `HonestDiD` |
| Partial identification / bounds | Frake et al. 2025 (SMJ) | Strategic management | What management currently adopts for unobserved-confounding robustness |

The key conceptual point for the paper's positioning: **negative-control *correction* and these *sensitivity* tools answer different questions.** NCO/NCE correction attempts a *point* (or set) estimate *net of* confounding under strong identifying assumptions; sensitivity analysis asks *how much* confounding it would take to overturn a result, under weak assumptions. The paper's Phase III→IV move (correct, then triangulate the corrected estimate against FE, IV, and these bounds) is exactly the right way to use them together — the corrected estimate is only as credible as its untestable assumptions, and the bounds tell you whether the qualitative conclusion survives their failure.

**The gap claim — assessment: defensible, and best framed as translation + integration, not de novo methodology.** No verified source integrates *detect → correct → triangulate* for panel fixed-effects designs in management/OM. The closest existing integration (the epidemiology scoping review) is a *method taxonomy*, not a *workflow*, and it is not panel-FE-specific, not management-facing, and has no triangulation/reporting-checklist layer. On the OM side, the causal-inference canon (MSOM OM Forum; Ketokivi & McIntosh 2017; Lu et al. 2018) contains no negative-control/proximal component at all, and management's current unobserved-confounding frontier is partial identification (Frake et al. 2025), not negative controls. So the paper's contribution is precisely: **(a)** translate the epidemiology/biostatistics negative-control apparatus into panel-FE OM language; **(b)** integrate detection, correction, and triangulation into a single decision workflow; **(c)** add the reporting-checklist and equivalence-bound discipline. That is a real and precise gap. **Caveat to state honestly in the paper:** because all verified correction-side evidence sits in epi/biostat/ML venues, the *absence* of an OM integration is consistent with the claimed gap but is not positive proof; the paper should frame novelty as "no integrated panel-FE workflow exists in management research," which the evidence supports, rather than "negative controls are new," which they are not — and should cite Ying et al. (2023) as the nearest longitudinal-proximal precedent it builds on.

---

## 6. Revision checklist for the paper

**Fix / confirm**
- [ ] Move the **additive equi-confounding** attribution from Tchetgen Tchetgen (2014) to **Sofer et al. (2016)**; keep Tchetgen Tchetgen (2014) for COCA/calibration generally (Phase III + Fig. caption).
- [ ] Confirm **"Lu et al. 2018" = Lu, Ding, Peng & Chuang (2018), JOM 64(1):53–64**; remove any "Lu & White 2014" placeholder (Phase IV, node 5).
- [ ] For the **proximal-2SLS** claim, cite Liu et al. (2025) as the regression/GLM implementation *plus* the proximal-causal-learning origin (Tchetgen Tchetgen et al. 2020; Cui et al. 2024) — not Liu et al. (2025) alone (Phase III, node 4b).
- [ ] Reframe **Danieli et al. (2026)** as an IV-design *caution* (false positives; functional-form) motivating equivalence bounds, not an endorsement of NCO detection (Phase II).
- [ ] Remove any implication that **Miao et al. (2018)** grounds the "one proxy = detect / two proxies = correct" split (refuted); re-anchor to Lipsitch (2010) / Miao (2018) + Tchetgen Tchetgen (2014).

**Add**
- [ ] **Hartman & Hidalgo (2018, AJPS)** — the equivalence-testing origin of "never a bare p > 0.10" (Phase II).
- [ ] **Roth (2022, AER: Insights)** — pre-test power and pre-testing bias in the panel-FE/DiD setting (Phase II detection-power argument).
- [ ] **Ying, Miao, Shi & Tchetgen Tchetgen (2023, JRSS-B)** — nearest longitudinal/panel proximal extension; cite and distinguish (Phase III + gap claim).
- [ ] **Rambachan & Roth (2023, REStud)** honest DiD and **VanderWeele & Ding (2017)** E-value — FE-native and cross-disciplinary sensitivity benchmarks (Phase IV).
- [ ] A **relevance/weak-proxy diagnostic** and an **invalid-proxy robustness** discussion for proximal 2SLS (Phase III; cite the 2025 invalid/weak-proxy cluster).
- [ ] A **multiple-testing** policy for nominating/testing many NCO candidates (Phase I→II) — no existing negative-control source covers this; owning it is a contribution.

**Positioning language**
- [ ] Frame novelty as *"first integrated detect→correct→triangulate workflow for panel-FE designs in empirical OM/management,"* explicitly acknowledging the mature epidemiology apparatus and Ying et al. (2023) as the nearest longitudinal precedent — a translation-and-integration claim, which the evidence supports, rather than a de-novo-method claim, which it does not.

---

## 7. References (verified)

**Negative-control foundations & detection**
- Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology* 21(3):383–388. https://pubmed.ncbi.nlm.nih.gov/20335814/
- Shi, X., Miao, W., & Tchetgen Tchetgen, E. (2020). A Selective Review of Negative Control Methods in Epidemiology. *Current Epidemiology Reports* 7(4):190–202. https://pubmed.ncbi.nlm.nih.gov/33996381/
- Yang, J., et al. (2023/24). Advances in methodologies of negative controls: a scoping review. *J. Clinical Epidemiology*. https://www.jclinepi.com/article/S0895-4356(23)00318-9/fulltext
- Duke-Margolis (2023). Understanding the Use of Negative Controls to Assess the Validity of Non-Interventional Studies (workshop deck). https://healthpolicy.duke.edu/sites/default/files/2023-03/NegativeControlWorkshopSlideDeck.pdf

**Placebo/falsification critiques (Phase II)**
- Eggers, A. C., Tuñón, G., & Dafoe, A. (2024). Placebo Tests for Causal Inference. *AJPS* 68(3):1106–1121. https://onlinelibrary.wiley.com/doi/full/10.1111/ajps.12818
- Hartman, E., & Hidalgo, F. D. (2018). An Equivalence Approach to Balance and Placebo Tests. *AJPS* 62(4):1000–1013. https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12387
- Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *AER: Insights* 4(3):305–322. https://www.aeaweb.org/articles?id=10.1257/aeri.20210236
- Danieli, O., Nevo, A., Walk, I., Weinstein, B., & Zeltzer, D. (2026). Negative Control Falsification Tests for Instrumental Variable Designs. *American Economic Review* 116(4):1380–1414. https://www.aeaweb.org/articles?id=10.1257/aer.20240636 (preprint https://arxiv.org/abs/2312.15624)
- Hussain, Z., et al. (2022). Falsification before Extrapolation in Causal Effect Estimation. *NeurIPS 2022*. https://proceedings.neurips.cc/paper_files/paper/2022/file/28b5dfc51e5ae12d84fb7c6172a00df4-Paper-Conference.pdf

**Correction: calibration & DiD (Phase III)**
- Tchetgen Tchetgen, E. J. (2014). The Control Outcome Calibration Approach for Causal Inference With Unobserved Confounding. *Am. J. Epidemiology* 179(5):633–640. https://pmc.ncbi.nlm.nih.gov/articles/PMC3927977/
- Sofer, T., Richardson, D. B., Colicino, E., Schwartz, J., & Tchetgen Tchetgen, E. J. (2016). On Negative Outcome Control of Unobserved Confounding as a Generalization of Difference-in-Differences. *Statistical Science* 31(3):348–361. https://pmc.ncbi.nlm.nih.gov/articles/PMC5322866/

**Correction: proximal inference (Phase III)**
- Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika* 105(4):987–993. https://academic.oup.com/biomet/article-abstract/105/4/987/5073056
- Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., & Miao, W. (2020). An Introduction to Proximal Causal Learning. arXiv:2009.10982 (expanded in *Statistical Science*, 2024). https://arxiv.org/abs/2009.10982
- Ying, A., Miao, W., Shi, X., & Tchetgen Tchetgen, E. J. (2023). Proximal Causal Inference for Complex Longitudinal Studies. *JRSS-B* 85(3):684–704. https://academic.oup.com/jrsssb/article/85/3/684/7094061
- Liu, J., Park, S., Li, K., & Tchetgen Tchetgen, E. J. (2025). Regression-Based Proximal Causal Inference. *Am. J. Epidemiology* 194(7):2030–2036. https://academic.oup.com/aje/article/194/7/2030/7775568
- Adaptive Proximal Causal Inference with Some Invalid Proxies (2025). arXiv:2507.19623. https://arxiv.org/abs/2507.19623

**Data-driven discovery (Phase III)**
- Kummerfeld, E., Lim, S., & Shi, X. (2024). Data-driven Automated Negative Control Estimation (DANCE). *JMLR* 25:1–35. https://jmlr.org/papers/volume25/22-1062/22-1062.pdf

**Sensitivity / triangulation (Phase IV)**
- Oster, E. (2019). Unobservable Selection and Coefficient Stability. *JBES* 37(2):187–204. https://www.tandfonline.com/doi/abs/10.1080/07350015.2016.1227711
- Cinelli, C., & Hazlett, C. (2020). Making Sense of Sensitivity: Extending Omitted Variable Bias. *JRSS-B* 82(1):39–67. https://academic.oup.com/jrsssb/article/82/1/39/7056023
- Busenbark, J. R., Yoon, H., Gamache, D. L., & Withers, M. C. (2022). Omitted Variable Bias: Examining Management Research With the ITCV. *J. Management* 48(1):17–48. https://journals.sagepub.com/doi/10.1177/01492063211006458
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Ann. Intern. Med.* 167(4):268–274. https://www.acpjournals.org/doi/10.7326/M16-2607
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies* 90(5):2555–2591. https://academic.oup.com/restud/article-abstract/90/5/2555/7039335

**OM / management endogeneity & adoption context**
- Ketokivi, M., & McIntosh, C. N. (2017). Addressing the endogeneity dilemma in operations management research. *J. Operations Management* 52:1–14. https://harisportal.hanken.fi/en/publications/addressing-the-endogeneity-dilemma-in-operations-management-resea
- Lu, G., Ding, X. (David), Peng, D. X., & Chuang, H. H.-C. (2018). Addressing endogeneity in operations management research: Recent developments, common problems, and directions for future research. *J. Operations Management* 64(1):53–64. https://onlinelibrary.wiley.com/doi/10.1016/j.jom.2018.10.001
- Frake, J., et al. (2025). From Perfect to Practical: Partial Identification Methods for Causal Inference in Strategic Management Research. *Strategic Management Journal*. https://sms.onlinelibrary.wiley.com/doi/full/10.1002/smj.3714
- OM Forum — Causal Inference Models in Operations Management. *M&SOM*. https://pubsonline.informs.org/doi/abs/10.1287/msom.2017.0659

---

*Verification note.* All primary claims above were confirmed against primary sources via a 3-vote adversarial verification pass (24/25 verified claims unanimous; 1 refuted, noted in §4). The two OM-side citations (Ketokivi & McIntosh 2017; Lu et al. 2018) and the "Lu et al. 2018" identity were verified by direct follow-up search. Where an attribution nuance exists it is flagged in §2; where the evidence is consistent-but-not-conclusive (the gap claim) it is labeled as such in §5. Citations the paper still owns independently: exact page/DOI details for any edition the author cites, and the final intended referents for "Lu et al. 2018" and the proximal-2SLS origin pairing.
