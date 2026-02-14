# PRODUCT.md

## Purpose

`objectless_alife` is a research-oriented product that explores whether structured, non-trivial dynamics can emerge in multi-agent systems without explicit objective functions.

## Target Users

- ALife and complex systems researchers
- Engineers prototyping open-ended simulation pipelines
- Technical collaborators analyzing phase-based observation designs

## Core Value Proposition

- Objective-free exploration with reproducible simulation behavior
- Side-by-side comparison across four observation models
- Analysis-ready output artifacts for downstream interpretation

## Key Features

- Seeded rule generation and deterministic simulation replayability
- Four-phase observation design:
  - Phase 1: local density only (20-entry table)
  - Phase 2: density + dominant neighbor state profile (100-entry table)
  - Phase 3 (Control): density + step-clock, no neighbor info (100-entry table)
  - Phase 4 (Random Walk): single-entry table, no observations
- Physical inconsistency filtering (halt and state-uniform termination)
- Optional dynamic filters for ablation experiments
- Rich metric extraction and Parquet-based logging
- Animation rendering for qualitative behavior review
- Statistical significance testing (Mann-Whitney U, chi-squared, Holm-Bonferroni)

## Business/Research Objectives

- Validate feasibility of objective-free search under constrained world dynamics
- Quantify differences between phase designs using shared experimental protocol
- Produce reproducible artifacts suitable for further statistical analysis and publication decisions
- Publish findings at ALIFE conference

## Non-Goals

- No reward optimization or fitness-based selection in the core loop
- No production service/API layer at this stage
- No interactive UI requirement for primary workflow
