---
title: "Review: Emergent Spatial Coordination from Negative Selection Alone"
reviewer_role: "ALife researcher"
date: "2026-02-21"
recommendation: "Weak Accept (leaning Accept) — contingent on clarifications and tightening of claims"
---

## Summary

This paper studies whether spatial coordination can emerge in a multi-agent grid world under an **objective-free** search regime in which candidate rule tables are generated randomly and then filtered only by a **viability (negative selection) criterion** that rejects degenerate dynamics. The key experimental manipulation is **observation richness**: agents range from a random walk / step-clock control to density-only observation and a richer state-profile observation. The authors measure coordination primarily via **mutual information (MI)** between neighbor states, emphasizing **excess MI** computed relative to a shuffle null that controls for pair-count bias, and use a Miller–Madow bias correction. They report that the state-profile condition shows nonzero median excess MI, while density-only and controls remain near zero, and that an “evidence ladder” (Control ≤ Phase 1 < Phase 2) holds across comparisons. Moran’s $I$ is used as a complementary spatial statistic; supplementary material reportedly includes robustness checks (multi-seed for top rules, halt-window sensitivity, capacity-matched controls, transfer entropy).

Overall, the paper is clearly motivated, sits squarely in ALife’s long-running discussion of open-endedness and evaluation bias, and makes a plausible and interesting claim: **observation channels can be sufficient to produce coordination even without a positive objective**. The main contribution is not a new model per se, but a careful *measurement* and *null-model* framing for coordination in rule-table systems subjected to minimal filtering.

## Strengths

1. **Timely conceptual framing**: The “no positive objective × viability-only selection” quadrant is a useful axis for organizing work on open-endedness, novelty search, and minimal criteria / minimal constraints approaches.
2. **Strong attention to measurement bias**: Explicitly addressing pair-count bias (and separating raw MI from excess MI) is excellent and should be of broad interest for ALife work that uses local statistics.
3. **Ablation via observation richness**: The condition design aligns with the main hypothesis and produces an interpretable ordering.
4. **Scale**: Evaluating 5,000 rules per condition is substantial for a rule-table search study and helps avoid cherry-picking.
5. **Transparency**: The manuscript points to a reproducibility map and an archive DOI; the supplementary scope sounds thorough (seed robustness, sensitivity analyses, and controls).

## Major concerns / required clarifications

### 1) “Objective-free” vs “minimal criteria” vs “selection pressure”
The paper appropriately notes that viability filtering is a minimal selection pressure, but the narrative still risks over-claiming “no fitness / no objective” in a way that may be interpreted as “no evaluation-driven bias at all.” Viability criteria can impose **strong, structured constraints** that shape the distribution of surviving rules.

*Requested action:* Please sharpen terminology:
- Define the viability criterion in a way that makes clear **what behaviors are explicitly disallowed** and how that implicitly biases toward particular dynamical regimes.
- Discuss relation to “minimal criteria” and to work where the criterion is deliberately weak but still shapes search (even if you are not doing adaptive evolution here).

### 2) Viability filter definition and edge cases
The introduction states that rules producing “all agents halt or converge to a single state” are removed. Later there is mention of “10-step halt window.” It is not yet fully clear (from the excerpt available in the main text) how viability is operationalized over time and across densities.

*Requested action:* Put the viability filter in one crisp block (pseudo-code or a numbered list) in the main paper (not only supplement), including:
- How “halt” is detected (no state changes? no movement?).
- What constitutes “converge to a single state” (global homogeneity? local?).
- Whether filters are applied uniformly across densities and observation conditions.
- Whether the filter is applied after a fixed burn-in / observation window.
- How ties/thresholds are handled (e.g., partial halting).

### 3) MI estimator details and comparability across rules
The paper says a Miller–Madow bias-corrected estimator is used throughout and that excess MI uses a shuffle null with $N=200$ shuffles (with a reported noise floor ~0.050 bits). This is promising, but the paper should ensure readers can reproduce the statistic exactly.

*Requested action:* Clarify:
- The exact random variables for MI (e.g., neighbor-pair state at time $t$? aggregated across time steps? across all neighbor pairs?).
- Whether you compute MI per step then average, or aggregate counts then compute once.
- Whether MI is computed conditionally on density or with density as a covariate (likely not, but say so).
- The definition of “pair-count bias” in your context (e.g., when there are more neighbor pairs at higher density, raw MI can increase even with independence).
- How the shuffle null is constructed (shuffle what exactly: agent labels, spatial positions, time steps, within-step permutations, etc.).
- Whether Miller–Madow is applied before/after excessing; and whether the shuffle null uses the same correction.

### 4) Interpretation of MI as “coordination”
MI between neighbor states is a reasonable proxy for local statistical dependence, but MI alone does not distinguish:
- coordinated patterns vs frozen patterns (though you filter out some degenerate dynamics),
- local couplings that arise from trivial interaction mechanics,
- spatial clustering vs genuine reciprocal coordination.

You mention Moran’s $I$ and transfer entropy in the supplement; that helps. However, the main text should be careful not to equate MI with “meaningful” coordination without more interpretive guardrails.

*Requested action:* In the main paper, add a short section explicitly articulating what MI does/doesn’t imply, and include at least one qualitative depiction (already likely in figures) tied to the MI regimes:
- Example snapshots / spacetime diagrams for (i) high MI Phase 2, (ii) near-zero MI Phase 1, (iii) random walk raw MI inflated but excess MI ~0.

### 5) Single-seed evaluation in the main experiment
You state that each rule table is evaluated with a single initial configuration in the main experiment, with multi-seed checks for the top 50 rules per condition in the supplement. This is defensible but could still bias condition-level conclusions if the seed distribution interacts with observation richness (especially when comparing medians across thousands of rules).

*Requested action:* Provide (in the main text) at least a brief summary statistic of seed robustness beyond “top 50,” e.g.:
- correlation of MI across seeds for a random subset of rules per condition,
- or a small $n$ (say 100 rules) multi-seed experiment sufficient to show the median ordering is not seed-artifactual.

## Minor comments / suggestions

1. **Effect size reporting**: You report Cliff’s $\delta$ and a bootstrap CI for a median difference, which is great. Consider also reporting an interpretable proportion metric (e.g., $P(\mathrm{MI}_{excess}>0)$) per condition.
2. **Terminology**: “Phase 1” / “Phase 2” is somewhat opaque without a reminder (density-only vs state-profile). Consider naming conditions consistently throughout.
3. **Evidence ladder**: Explicitly list the comparisons included and whether they are planned contrasts; clarify multiple-comparison handling if relevant.
4. **Noise floor**: If 0.050 bits is the noise floor at $N=200$ shuffles, confirm how this scales with more shuffles or with rule length / number of samples. Consider a brief convergence plot in supplement.
5. **Capacity-matched controls**: This is an important confound and should be highlighted more in the main text if space allows, as it directly supports the key claim.
6. **Reproducibility pointers**: The reproducibility map reference is good; in the paper, ensure that the artifact lineage is stable and that the archive contains the exact commit used for results (even if anonymized).
7. **Relation to prior work**: You cite Stanley & Lehman-style open-endedness and novelty search; it may help to cite minimal criteria / viability-based selection and self-organization / collective behavior work more explicitly, to position MI-based coordination among classic ALife pattern formation literature.

## Questions for the authors

1. How sensitive are the MI results to the neighborhood definition (von Neumann vs Moore), grid size, and number of states?
2. Can you characterize the high-MI Phase 2 rule tables structurally (e.g., prevalence of particular transitions), beyond ranking by MI?
3. Does the viability filter disproportionately remove rules in some conditions (e.g., richer observation leading to more/less survival)? If so, how does that affect interpretation of “observation richness drives coordination”?
4. Is there a regime where density-only observation yields coordination if you change the rule-table capacity or allow longer temporal memory? That would help disentangle observation richness from memory / state capacity.

## Overall evaluation

The paper is a solid contribution with strong measurement hygiene and a clear, testable hypothesis. With clearer definitions of the viability filter and MI/shuffle procedure, a slightly more cautious interpretation of “objective-free,” and a small additional seed-robustness summary in the main text, I would be comfortable recommending acceptance.
