## What

<!-- One or two sentences. -->

## Issue

<!-- Into main:            Closes #<issue>
     Into an issue branch: Part of #<issue>. Resolves #<sub-issue> (close it manually after merge) -->

## Type

- [ ] Sub-issue → issue branch (squash merge)
- [ ] Issue → main (merge commit)

## How it was tested

<!-- Commands run and their results. Say "not run" if something was not run. -->

## Metrics (required for detector / grouping / ml changes)

| Metric | Baseline | This PR |
|---|---|---|
| Detection precision / recall / mAP@0.5 | | |
| Grouping ARI / NMI (ground-truth boxes) | | |
| Latency p50 / p95 | | |

Workload and machine:

## Checklist

- [ ] Acceptance criteria met
- [ ] Tests added/updated; CI green
- [ ] Docs updated (README / docs / experiment report / model card)
- [ ] `.env.example` updated if config changed
- [ ] CHANGELOG `Unreleased` updated (PRs into main)
- [ ] No datasets, weights, secrets or large binaries committed
- [ ] Known limitations listed below

## Limitations / follow-ups
