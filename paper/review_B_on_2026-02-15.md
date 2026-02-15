# Peer Review (ALife)

**Manuscript title:** *Emergent Spatial Coordination from Negative Selection Alone: The Role of Observation Richness in Objective-Free Artificial Life*  
**Reviewer perspective:** Artificial Life / complex systems / multi-agent systems

## Summary
The paper investigates whether spatial coordination can emerge in a multi-agent grid world under an “objective-free” search regime: many random local rules are generated and only “physically inconsistent” rules are filtered out (negative selection). The central claim is that richer observation channels (especially “state-profile observation”) enable nonzero coordination measured by a permutation-calibrated “excess mutual information” (MI), whereas lower-information observations (density-only, step-clock control, random walk) do not. The study reports large-scale sweeps (e.g., thousands of rules per condition), uses a Miller–Madow bias-corrected MI estimator, introduces a shuffle null to control pair-count bias, and supplements with Moran’s $I$ to separate local coordination from global clustering.

Overall, the work addresses an important ALife question: what minimal ingredients support emergent coordination when the usual “fitness shaping” is removed. The emphasis on measurement bias (pair counts, estimator bias) and explicit nulls is a strength. However, several core concepts (negative selection definition, “physical inconsistency,” rule parameterization, and observation regimes) need sharper formalization, and the evaluation pipeline needs additional controls and ablations to convincingly support the causal claim that *observation richness per se* is the driver.

## Major strengths
1. **Timely, conceptually interesting question (open-endedness / objective-free ALife).** The framing directly engages ongoing concerns about evaluation bias and implicit objectives.
2. **Attention to statistical/estimation pitfalls.** Using a shuffle null to calibrate MI and correcting for small-sample bias (Miller–Madow) is good practice; many ALife papers would stop at raw MI.
3. **Scale and robustness intent.** Reporting thousands of rules and a density sweep indicates a serious attempt to avoid cherry-picking.
4. **Multiple measures of structure.** The addition of Moran’s $I$ as a complementary spatial statistic is appropriate and helps disambiguate pattern types.

## Major concerns / required revisions

### 1) Clarify and formalize “negative selection” and “physical inconsistency”
The manuscript states that rules are filtered if they produce “trivially broken simulations (all agents halt or converge to a single state).” This is underspecified and risks being an implicit objective.

**What to add:**
- A precise operational definition of *physical inconsistency*, ideally as an algorithm with thresholds (e.g., “halt” definition, time horizon, convergence criterion, tolerance, how many seeds must fail).
- The fraction of rules discarded per condition and whether discard rates differ substantially across observation regimes (if they do, that itself could explain downstream differences).
- Evidence that the filter does not inadvertently select for coordination (e.g., by selecting for dynamical diversity or sustained activity, which often correlates with MI).

**Suggested analysis:** report MI (and other metrics) **before** and **after** filtering, or at least show that the filter is independent of the coordination metric (e.g., correlation between survival/consistency and excess MI; or a stratified analysis at equal survival bands).

### 2) Define the rule space and observation/action model unambiguously
At present, key parts of the model appear only at a high level (from the abstract and glimpses): “random rule generation,” “step-clock control,” “density-only observation,” and “state-profile observation.”

**What to add (minimum for reproducibility and interpretability):**
- The agent state space size $|S|$ and whether states are discrete, plus whether there is a “halt” state.
- The action space: is movement allowed? collisions? asynchronous vs synchronous updates? boundary conditions? (torus vs walls).
- The observation encoding for each condition in explicit terms: e.g., for state-profile, is the observation a vector of neighbor states, a histogram, an ordered tuple, or something else? What neighborhood (von Neumann, Moore, radius)?
- The rule table structure: mapping from observation to next state/action; whether stochasticity exists; and what is randomized (entire table i.i.d.? constraints?).
- The simulation horizon $T$ used for measurement and filtering.

Without these, it is difficult to assess whether the “richer observation” condition also implicitly increases effective rule-table capacity or changes the dynamics in other ways.

### 3) Separate “observation richness” from “policy capacity” and “effective branching factor”
The main causal claim is that observation-channel richness drives coordination “not rule table capacity or selection pressure.” But richer observations usually enlarge the conditional space, which increases the number of distinct situation-action mappings a rule can encode. Even if the table is fixed-size, the *effective* expressivity can differ.

**Required controls/ablations:**
- **Capacity-matched control:** hold the number of observation symbols constant across conditions (e.g., compress state-profile to match density-only alphabet size) or constrain policies to use a fixed number of parameters (e.g., decision trees of fixed depth), then vary observation richness.
- **Information-matched control:** explicitly equalize mutual information between the observation and the true neighbor configuration (e.g., through randomized encoding) to test whether it is “richness” vs “structure in the encoding.”
- **Random encoding control:** keep the cardinality of the observation alphabet the same as state-profile but permute the mapping from true neighborhood states to observation symbol; if performance drops, the effect is about semantic alignment rather than alphabet size.

### 4) Statistical reporting needs tightening (effect sizes, CIs, and consistency)
The abstract reports: “Cliff’s $\delta = 0.34$; median-difference bootstrap 95% CI $[0.26, 0.29]$.” A confidence interval that does not contain the point estimate and is extremely narrow relative to the estimate is suspicious; perhaps the CI is for something else (or a typo).

**Required fixes:**
- Verify and correct all effect-size/CI pairings; specify what the CI is for (median difference? Cliff’s $\delta$?).
- Report sample sizes used for each test (rules after filtering, number of seeds, number of neighbor pairs).
- Provide exact $p$-values (or at least thresholds) after Holm–Bonferroni, and clarify whether tests are on per-rule medians, per-seed values, or pooled pairs (pooling can inflate $n$).
- Clearly separate *within-rule* variability (across seeds) from *across-rule* variability.

### 5) The MI pipeline: define “excess MI” precisely and validate the null model
The idea of “excess MI” calibrated by a permutation-based shuffle null is promising, but the null must preserve the right nuisance structure.

**What to specify/validate:**
- What is being shuffled (agent labels? time indices? neighbor pairs?) and what is preserved (marginals, temporal autocorrelation, spatial adjacency)?
- Is the shuffle done within each timestep, across all timesteps, or across the whole run?
- How many shuffles per rule/seed? How stable is the estimate?
- Why does the null control “pair-count bias,” and does it also control for marginal state imbalance (which can inflate MI)?

**Recommended additions:**
- Compare the shuffle null to at least one analytic baseline or alternative null (e.g., fixed marginals with independent sampling; block-shuffle to preserve temporal correlations).
- Show that random walk has elevated raw MI but near-zero excess MI across densities and horizons (the abstract claims this; include a figure or table).

### 6) Interpretation: distinguish coordination from shared environment / common-cause effects
Neighbor MI can arise from shared constraints (crowding, movement rules, boundary effects, update synchronicity) rather than active coordination. Moran’s $I$ helps, but more is needed.

**Suggested tests:**
- **Causal directionality:** compute transfer entropy or conditional MI (e.g., $I(X_t; Y_{t+1} \mid Y_t)$) to test whether neighbor states predict future states beyond self-history.
- **Spatial scrambling control:** keep global state counts per timestep but randomize agent positions; does MI drop?
- **Interaction removal:** run the same rules in a version where agents do not observe neighbors (or observations are replaced by noise) while keeping movement/physics identical; quantify difference.

## Minor comments / editorial
1. **Terminology:** “objective-free” should be defined carefully; negative selection is still a selection mechanism. Consider “no positive objective / viability-only selection” or “viability filtering.”
2. **Evidence ladder phrasing:** “Control $\leq$ Phase 1 $<$ Phase 2” is suggestive but should be backed by formal comparisons and explicitly stated metrics.
3. **Figures/tables:** ensure that every key claim in the abstract has a direct pointer to a figure/table in the main text (not only appendix).
4. **Reproducibility:** include pseudocode of rule generation, filtering, simulation loop, and MI estimation/shuffle in an appendix; provide all hyperparameters.
5. **Units and calibration:** state whether MI is in bits (it appears so) and confirm consistent log base.

## Questions for the authors
1. How sensitive are the results to simulation horizon $T$? Does Phase 2’s excess MI emerge early and persist, or only transiently?
2. How does the survival/filtering rate differ by condition, and does excess MI remain when conditioning on survival?
3. What is the minimal observation richness that yields nonzero median excess MI? (e.g., can a 1-bit neighbor feature work?)
4. Are there qualitative classes of emergent patterns (e.g., stripes, domains, oscillatory waves)? Can you show representative space-time plots for top rules and median rules?
5. Can you demonstrate that Phase 2 rules generalize across densities/grid sizes beyond the tested sweep (e.g., extrapolation)?

## Overall recommendation
**Weak accept / major revision (depending on venue bar).** The premise and measurement care are strong, but the paper needs tighter formal definitions, capacity-matched controls, and clearer statistical reporting to support the central causal claim.

---

## Detailed recommendations (actionable checklist)
Below is a concrete set of changes that would make the paper substantially stronger and easier to evaluate/reproduce.

### A. Model specification & reproducibility (highest priority)
1. **Add a full model spec table** (one place) listing: grid size, boundary conditions, neighborhood type, synchrony, collision/occupancy rules, number of discrete states $|S|$, presence of a halting state, update schedule, simulation horizon $T$, number of seeds, and any randomization details.
2. **Define each observation channel as an explicit mapping** from the true neighborhood configuration to an observation symbol (or vector), including the observation alphabet size for each condition.
3. **Define the rule table structure and sampling scheme**: deterministic vs stochastic policy, mapping $(\text{obs}, \text{self state}) \to (\text{action}, \text{next state})$ (or whatever is used), and how entries are randomized (i.i.d., constrained, symmetric, etc.).
4. **Add pseudocode** for (i) rule generation, (ii) simulation loop, (iii) filtering/negative selection, (iv) metric computation and null calibration. (Appendix is fine.)

### B. Negative selection / “physical inconsistency” (very high priority)
5. **Operationalize physical inconsistency** as a deterministic procedure with thresholds (e.g., “halt” definition, “converge to a single state” definition, time window, tolerance).
6. **Report discard/survival rates per condition** (and per density in the sweep), with uncertainty over seeds.
7. **Demonstrate filter–metric independence**:
   - correlation plots between survival/consistency and excess MI;
   - stratify comparisons at matched survival (e.g., compare Phase~1 vs Phase~2 among rules within the same survival quantiles);
   - optionally report “all rules” vs “post-filter” results to show what filtering changes.

### C. Controls that isolate “observation richness” from capacity/encoding (very high priority)
8. **Capacity-matched policy control**: repeat the main comparison with a policy class that has fixed parameter budget across conditions (e.g., decision tree of fixed depth, fixed-size neural net, or hashed table with collisions), while varying observation richness.
9. **Random encoding control**: keep the same observation alphabet size as state-profile but randomly permute the encoding from neighborhood configurations to observation symbols; show whether coordination collapses.
10. **Information-matched control**: compress state-profile observations to match the mutual information (or entropy) of density-only observations; test whether performance tracks information content rather than representation.

### D. Metric definition, null models, and robustness (high priority)
11. **Precisely define excess MI** with an equation: e.g., $\mathrm{MI}_\mathrm{excess} = \mathrm{MI}_\mathrm{obs} - \mathbb{E}[\mathrm{MI}_\mathrm{shuffle}]$ (or whatever is used), including the shuffle operator and what statistics are preserved.
12. **Validate the null model** by adding at least one alternative null:
    - within-timestep shuffle vs across-time shuffle;
    - block-shuffle preserving temporal autocorrelation;
    - fixed-marginal independent baseline.
13. **Clarify the unit of analysis** for statistics (per-rule aggregated over seeds vs per-seed), and avoid pooling neighbor-pairs as independent samples.
14. **Robustness to horizon $T$**: report excess MI time series or at multiple $T$ values (early/mid/late), and show whether the effect is transient or stable.
15. **Robustness to density and grid size**: integrate the density sweep into the main argument (not only appendix), emphasizing where the effect turns on/off.

### E. Distinguish coordination from common-cause structure (medium–high priority)
16. **Spatial scrambling control**: preserve global state counts but randomly permute agent positions each timestep; MI that survives this is likely not local coordination.
17. **Directional dependence**: add a causal-ish statistic (transfer entropy or conditional MI) to test whether neighbor states predict future states beyond self-history.
18. **No-interaction baseline**: rerun the same rule tables with neighbor observation replaced by noise (keeping physics identical) to quantify the direct contribution of interaction information.

### F. Presentation & statistical reporting (medium priority)
19. **Fix/verify effect size + CI reporting** (the Cliff’s $\delta$ vs CI mismatch in the abstract must be corrected); include $n$ for each test and show corrected $p$-values.
20. **Add representative qualitative examples**: space-time plots for a few “typical” rules (median) and a few “top” rules in each condition, with the same visualization pipeline.
21. **Clarify claims**: avoid wording that implies proof of causality; instead state that results are “consistent with” observation richness as a driver, then point to the controls above as causal support.

---

## Score (if the above recommendations are implemented)
Assuming the paper currently sits around a **6/10** (strong idea, incomplete specification/controls), then **if all items A–F are implemented well and results remain consistent**, I would expect a score around **8.5/10**.

**Rationale for 8.5/10:** the contribution would shift from “promising result with careful MI calibration” to a **well-controlled, reproducible** demonstration that isolates observation richness from confounds (capacity/encoding/filtering), with robust nulls and stronger evidence that the measured MI reflects genuine interaction-driven coordination.

---

## Detailed recommendations (actionable checklist)
Below is a concrete set of changes that would make the paper substantially stronger and easier to evaluate/reproduce.

### A. Model specification & reproducibility (highest priority)
1. **Add a full model spec table** (one place) listing: grid size, boundary conditions, neighborhood type, synchrony, collision/occupancy rules, number of discrete states $|S|$, presence of a halting state, update schedule, simulation horizon $T$, number of seeds, and any randomization details.
2. **Define each observation channel as an explicit mapping** from the true neighborhood configuration to an observation symbol (or vector), including the observation alphabet size for each condition.
3. **Define the rule table structure and sampling scheme**: deterministic vs stochastic policy, mapping $(\text{obs}, \text{self state}) \to (\text{action}, \text{next state})$ (or whatever is used), and how entries are randomized (i.i.d., constrained, symmetric, etc.).
4. **Add pseudocode** for (i) rule generation, (ii) simulation loop, (iii) filtering/negative selection, (iv) metric computation and null calibration. (Appendix is fine.)

### B. Negative selection / “physical inconsistency” (very high priority)
5. **Operationalize physical inconsistency** as a deterministic procedure with thresholds (e.g., “halt” definition, “converge to a single state” definition, time window, tolerance).
6. **Report discard/survival rates per condition** (and per density in the sweep), with uncertainty over seeds.
7. **Demonstrate filter–metric independence**:
   - correlation plots between survival/consistency and excess MI;
   - stratify comparisons at matched survival (e.g., compare Phase~1 vs Phase~2 among rules within the same survival quantiles);
   - optionally report “all rules” vs “post-filter” results to show what filtering changes.

### C. Controls that isolate “observation richness” from capacity/encoding (very high priority)
8. **Capacity-matched policy control**: repeat the main comparison with a policy class that has fixed parameter budget across conditions (e.g., decision tree of fixed depth, fixed-size neural net, or hashed table with collisions), while varying observation richness.
9. **Random encoding control**: keep the same observation alphabet size as state-profile but randomly permute the encoding from neighborhood configurations to observation symbols; show whether coordination collapses.
10. **Information-matched control**: compress state-profile observations to match the mutual information (or entropy) of density-only observations; test whether performance tracks information content rather than representation.

### D. Metric definition, null models, and robustness (high priority)
11. **Precisely define excess MI** with an equation: e.g., $\mathrm{MI}_\mathrm{excess} = \mathrm{MI}_\mathrm{obs} - \mathbb{E}[\mathrm{MI}_\mathrm{shuffle}]$ (or whatever is used), including the shuffle operator and what statistics are preserved.
12. **Validate the null model** by adding at least one alternative null:
    - within-timestep shuffle vs across-time shuffle;
    - block-shuffle preserving temporal autocorrelation;
    - fixed-marginal independent baseline.
13. **Clarify the unit of analysis** for statistics (per-rule aggregated over seeds vs per-seed), and avoid pooling neighbor-pairs as independent samples.
14. **Robustness to horizon $T$**: report excess MI time series or at multiple $T$ values (early/mid/late), and show whether the effect is transient or stable.
15. **Robustness to density and grid size**: integrate the density sweep into the main argument (not only appendix), emphasizing where the effect turns on/off.

### E. Distinguish coordination from common-cause structure (medium–high priority)
16. **Spatial scrambling control**: preserve global state counts but randomly permute agent positions each timestep; MI that survives this is likely not local coordination.
17. **Directional dependence**: add a causal-ish statistic (transfer entropy or conditional MI) to test whether neighbor states predict future states beyond self-history.
18. **No-interaction baseline**: rerun the same rule tables with neighbor observation replaced by noise (keeping physics identical) to quantify the direct contribution of interaction information.

### F. Presentation & statistical reporting (medium priority)
19. **Fix/verify effect size + CI reporting** (the Cliff’s $\delta$ vs CI mismatch in the abstract must be corrected); include $n$ for each test and show corrected $p$-values.
20. **Add representative qualitative examples**: space-time plots for a few “typical” rules (median) and a few “top” rules in each condition, with the same visualization pipeline.
21. **Clarify claims**: avoid wording that implies proof of causality; instead state that results are “consistent with” observation richness as a driver, then point to the controls above as causal support.

---

## Score (if the above recommendations are implemented)
Assuming the paper currently sits around a **6/10** (strong idea, incomplete specification/controls), then **if all items A–F are implemented well and results remain consistent**, I would expect a score around **8.5/10**.

**Rationale for 8.5/10:** the contribution would shift from “promising result with careful MI calibration” to a **well-controlled, reproducible** demonstration that isolates observation richness from confounds (capacity/encoding/filtering), with robust nulls and stronger evidence that the measured MI reflects genuine interaction-driven coordination.

