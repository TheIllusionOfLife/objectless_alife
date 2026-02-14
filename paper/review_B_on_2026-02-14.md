# Peer review (ALife) — *Emergent Spatial Coordination from Negative Selection Alone*

## Summary of the paper’s claim
The manuscript proposes an “objective-free” ALife workflow: sample many random shared rule tables for agents on a toroidal grid, apply only **negative selection** for “physical inconsistency” (halt / uniform-state collapse), then characterize surviving rules **post hoc** via information-theoretic metrics. The central empirical claim is that **observation-channel richness** (Phase 2: dominant-neighbor-state) yields nonzero median neighbor mutual information (MI), whereas density-only (Phase 1) and controls remain at/near zero.

## Overall assessment (go / no-go)
**Go, but only with revisions.** The research direction is strong and ALife-relevant (evaluation bias, objective-free discovery). The current framing overstates neutrality: viability filters and MI-based interpretation need stronger construct validity and controls.

---

## Major comments (high priority)

### 1) “Objective-free” vs implicit objectives introduced by filters
The two “physical” filters (halt detection; uniform-state convergence) are not purely physics-like constraints—they impose **behavioral/phenotypic viability**:
- Rejecting uniform-state convergence is tantamount to requiring ongoing **state diversity**.
- Rejecting halting/absorbing regimes is tantamount to requiring ongoing **activity**.

**Requested revisions**
- Reframe as “objective-free with respect to coordination, with explicit viability constraints.”
- Quantify filter impact per condition by reporting metric distributions:
  - **before filtering**
  - after halt filter
  - after uniformity filter
  - after both
- Add ablations: run (i) no filters (record metrics at fixed horizon), (ii) halt-only, (iii) uniformity-only, (iv) both. Show the Phase 2 advantage persists.

### 2) Construct validity of neighbor MI as “coordination”
Neighbor MI is a reasonable signal, but it can be driven by artifacts (marginal shifts, isolation coding, adjacency/contact patterns). If MI is mostly reflecting state marginal structure or isolation/no-neighbor conditions, it is weaker evidence of “coordination” in the ALife sense.

**Requested revisions**
- Make the MI sampling procedure explicit and consistent across the paper:
  - computed at final step only vs aggregated across time steps
  - how adjacent pairs are counted (directed vs undirected edges; duplicates over time; burn-in)
- Report MI together with:
  - state entropy $H(S)$ and/or normalized MI (so “nonzero MI” isn’t just a marginal artifact)
  - a grid-structure metric less sensitive to marginals, e.g. assortativity/same-state neighbor fraction above random, Moran’s I, join-counts, cluster-size distribution
- Add “contact-conditioned” MI: compute MI using only pairs where both agents have $n>0$ (or remove the “no neighbors” symbol from relevant conditioning analyses).

### 3) Controls: table-size matching is good, but semantic neighbor-state coupling remains un-isolated
The step-clock control matches the **rule table size** (100), but Phase 2 adds a channel that directly couples to neighbor state semantics. Reviewers may argue the effect is expected and not yet cleanly attributed to “richness” rather than “semantically aligned coupling.”

**Requested revisions**
- Add at least one “scrambled semantics” control:
  - permute neighbor-state labels when forming $d$ (random permutation per rule), or
  - replace $d$ with a same-cardinality non-semantic neighbor-derived signal.
- Alternatively, include an observation channel with the same alphabet size as $d$ that should not facilitate alignment (e.g., hashed category), to separate “bits” from “meaningful coupling.”

### 4) Comparability under selection: survival differs by condition
If survival rates differ (the appendix suggests Phase 2 survives more often), then “survivors-only” comparisons can conflate **viability** and **coordination**. This is effectively conditioning on a collider.

**Requested revisions**
- Report results both:
  - **conditional on survival** (current)
  - **unconditional**, e.g., assign MI = 0 to filtered-out runs (or compute MI at termination time) and compare full distributions.
- Decompose the effect: is Phase 2’s higher MI due to higher survival probability, or higher MI among survivors?

### 5) Statistical reporting: extreme p-values; emphasize effects and uncertainty
p-values like $p < 10^{-280}$ are not very informative and may trigger reviewer skepticism. The paper will be stronger with effect sizes and uncertainty intervals.

**Requested revisions**
- Put effect size and confidence intervals first (e.g., rank-biserial $r$, Cliff’s delta; bootstrap CI for median difference).
- Use ECDFs/violins and explicitly address the mass at zero (two-part distribution: spike at 0 + continuous tail).

---

## Minor comments (still worth addressing)

### Writing / positioning
- The “unexplored quadrant” claim should be softened. Many systems are “no explicit fitness” but still have viability constraints and implicit selection.
- Define “spatial coordination” operationally early and justify MI as that operationalization.

### Methods clarity
- Update order: state whether the random sequential order is re-sampled every step.
- Clarify whether failed moves still count as an action (they should, but state it).
- The encoding $d=4$ for “no occupied neighbors” may inject a strong isolation signal; consider conditioning analyses on $n>0$.

### Reproducibility
- Specify rule-table sampling distribution (uniform over 9 actions per entry?) and any symmetry constraints.
- Add pseudocode for simulation + filtering + metric computation.

### Presentation
- The “evidence ladder” phrasing is nonstandard; consider “ordering”/“ranking” of conditions.
- The abstract could trim very small p-values in favor of a concise effect size statement.

---

## Suggested additional experiments (if you add only a few)
1. **Filter ablation**: none / halt-only / uniformity-only / both, with both conditional and unconditional analyses.
2. **Condition-on-contact MI**: MI computed only on edges with non-isolated agents, plus a companion clustering/assortativity metric.
3. **Permutation/scramble control** for the Phase 2 neighbor-state signal to isolate semantics from alphabet size.
4. **Time-resolved MI trajectories** to show emergence over time rather than a final-step artifact.

---

## Final recommendation
**Promising and worth pursuing.** With the requested revisions—especially filter ablations, unconditional analyses, and at least one additional coordination/control metric—the paper would read as a solid ALife contribution rather than an over-claimed result.
- The encoding $d=4$ for “no occupied neighbors” may inject a strong isolation signal; consider conditioning analyses on $n>0$.