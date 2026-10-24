---
name: ML experiment
about: A measured experiment that ends with a written decision
title: "[Mx.y] EXP: "
labels: ["type: experiment", "area: ml"]
---

**Branch:** `exp/<this-number>-<slug>` · **Report:** `docs/experiments/EXP-<nnn>-<slug>.md`

## Hypothesis

<!-- "Changing X will improve metric Y on the eval set because Z." -->

## Deciding metric

<!-- e.g. grouping ARI on ground-truth boxes; detection mAP@0.5; latency p95. Detection and grouping are always reported separately. -->

## Variables

| Variable | Values |
|---|---|
| | |

## Fixed

<!-- Eval set version, seed, hardware, everything else held constant. -->

## Acceptance criteria

- [ ] All variants run against the same eval set and baseline
- [ ] Results table and plots in the report
- [ ] Decision written: adopt / reject / needs follow-up
- [ ] If adopted: promotion issue opened
