# Development Workflow

How work moves from an idea to `main` in the Retail AI Pipeline. Every change follows this, including documentation and experiments.

## 1. Work hierarchy

```
Milestone
└── Issue                     one deliverable, one branch, one PR into main
    ├── task checklist        default: steps tracked as checkboxes in the issue
    └── Sub-issues            only for large issues (see rule below)
```

- **Milestone**: a goal. Three milestones end in a release (v0.2.0, v0.5.0, v1.0.0).
- **Issue**: a deliverable with acceptance criteria and a **task checklist**. One branch, one PR into `main`.
- **Sub-issue**: a real GitHub sub-issue with its own branch.

**When to split an issue into sub-issues:** only when it is **size L** *and* its parts can be reviewed and tested on their own. Everything else stays one branch with a checklist. Working solo, each extra branch and PR is overhead, so the bar for splitting is high. You can still split an issue mid-way if it turns out bigger than expected.

The roadmap lives in `scripts/github/roadmap.json` (rendered to `docs/roadmap/ROADMAP.md`). Status lives on [project board #6](https://github.com/users/firrexguptaji/projects/6).

## 2. Branch model

**Normal issue (checklist):**

```
main ──────────────────────────●──
  └── fix/15-grouping-edge-cases ──●──●──┘   PR: "Closes #15"
```

**Split issue (size L, with sub-issues):**

```
main ─────────────────────────────────────────────────●──────── (protected)
  └── test/19-test-framework ───●────────●────────────┘   PR: "Closes #19"
        ├── sub/19/20-unit-tests ─────────┘ (squash)
        └── sub/19/21-api-tests ──────────────────┘ (squash)
```

| Branch | Created from | Merges into | Name |
|---|---|---|---|
| Issue branch | `main` | `main` | `<type>/<issue#>-<slug>` |
| Sub-issue branch | its issue branch | its issue branch | `sub/<issue#>/<sub#>-<slug>` |

`<type>` comes from the issue's type label: `feat`, `fix`, `exp` (experiment), `test`, `chore`, `docs`.

Sub-issue branches use the separate `sub/` prefix because Git cannot have both a branch `fix/12-x` and a branch `fix/12-x/13-y`.

### Commands

```bash
# Normal issue: one branch, one PR (move card to "In progress")
git switch main && git pull
git switch -c fix/15-grouping-edge-cases
# ... commit, tick tasks in the issue ...
git push -u origin fix/15-grouping-edge-cases
gh pr create --base main --title "fix(grouping): handle 0/1 detections, normalise input (#15)" --body "Closes #15"

# Split issue: issue branch first
git switch main && git pull
git switch -c test/19-test-framework
git push -u origin test/19-test-framework

# ...then one branch per sub-issue, off the issue branch
git switch -c sub/19/20-unit-tests
git push -u origin sub/19/20-unit-tests
gh pr create --base test/19-test-framework --title "test: pytest setup and unit tests (#20)" \
  --body "Part of #19. Resolves #20."

# After the sub-issue PR is squash-merged
git switch test/19-test-framework && git pull

# Keep the issue branch current with main (merge, never rebase a shared branch)
git fetch origin && git merge origin/main

# All sub-issues merged and CI green: PR the issue into main
gh pr create --base main --title "test: test framework across services (#19)" --body "Closes #19"
```

Issue numbers above are examples; real numbers come from GitHub.

### Merge strategy

| PR | Strategy | Why |
|---|---|---|
| sub-issue → issue branch | **Squash** | One clean commit per sub-issue: `test: pytest setup and unit tests (#20)` |
| split issue → `main` | **Merge commit** | Keeps the sub-issue commits as readable history under one merge |
| normal issue → `main` | **Squash** | One commit per issue |

Delete branches after merging.

### Closing issues

GitHub's `Closes #n` only works for PRs merged into the default branch (`main`). So:

- **Sub-issues**: close manually right after the PR merges into the issue branch (the work is integrated and tested there). Set the card to *Done*.
- **Issues**: closed automatically by `Closes #n` in the PR into `main`.

## 3. Board flow

| Status | Means | Who moves it |
|---|---|---|
| Backlog | Planned, not refined | — |
| Ready | Acceptance criteria clear, dependencies done | Before starting |
| In progress | Branch exists | When you create the branch |
| In review | PR open, CI running | When the PR is opened |
| Done | Merged (sub-issue: into issue branch; issue: into `main`) | On merge |

Keep the board's work-in-progress limits. Finish before starting more.

## 4. Commits

[Conventional Commits](https://www.conventionalcommits.org/), scoped by area, issue number at the end:

```
feat(detector): batch sliding-window inference (#41)
fix(grouping): handle single detection (#15)
test(gateway): failure-path API tests (#19)
exp(ml): yolov8s 1024px fine-tune run (#33)
docs(eval): baseline report (#24)
ci: docker build cache (#28)
```

## 5. Pull requests

Every PR uses `.github/pull_request_template.md`. A PR is mergeable when:

- CI is green (lint, tests, Docker build, and the eval gate when model code changed)
- The checklist in the template is complete
- Model or pipeline changes include a **before/after metrics table** from the eval harness
- Docs are updated in the same PR, not later

## 6. ML experiments

Experiments are issues labelled `type: experiment` on `exp/` branches.

1. State the **hypothesis** and the metric that decides it in the issue before starting.
2. Work in a numbered notebook under `notebooks/` that imports from `ml/` (no copied service code).
3. Log runs (params, metrics, weights hash). Never commit datasets or weights.
4. Write `docs/experiments/EXP-<nnn>-<slug>.md` from `docs/experiments/TEMPLATE.md`.
5. The experiment is **done when the report has a decision**, including "no improvement, keep baseline".
6. Shipping a winning model is a separate *promotion* issue, so production changes are reviewed on their own.

Detection quality and grouping quality are always reported separately.

## 7. Documentation by change type

| Change | Update |
|---|---|
| API or behaviour | README, `docs/architecture.md`, API tests |
| Config / env var | `.env.example`, `docs/deployment.md` |
| Model or parameters | experiment report, model card in `docs/models/`, `eval/baseline.json` |
| Infra / CI | `docs/deployment.md` or `docs/ci-cd.md` |
| Any user-visible change | `CHANGELOG.md` → `Unreleased` |

## 8. Definition of Done

An issue is **Done** only when:

- [ ] Acceptance criteria are met
- [ ] Tests are added or updated and pass in CI
- [ ] Docs and CHANGELOG are updated
- [ ] It is merged into `main` by PR
- [ ] Remaining limitations are written in the PR or the issue

Use these words precisely in issues and PRs: **proposed → implemented → tested → merged → released (tagged) → deployed**. They are not interchangeable.

## 9. Releases

Three releases, each marking a real change in what the system can do:

| Release | After | Meaning |
|---|---|---|
| `v0.2.0` | M1 + M2 | Reliable and measured: no crashes, tests, baseline numbers |
| `v0.5.0` | M3 + M4 | CI/CD in place, improved models promoted |
| `v1.0.0` | M5 + M6 | Fast, observable, documented |

Steps:

1. Move the `Unreleased` section of `CHANGELOG.md` under the new version.
2. Tag `main`: `git tag -a v0.2.0 -m "Reliable and measured" && git push --tags`.
3. From M3 onward the release workflow publishes images, notes, weights and the eval report.
4. Close the finished milestones on GitHub.

## 10. Measuring model changes

- The eval set starts at 20 labelled images (M2) and grows to 40+ (M4.2) before model decisions are made.
- Every metric is reported with a 95% bootstrap confidence interval, from resampling images.
- Compare variants with a paired bootstrap on the same images. If the interval of the difference includes 0, report "no measurable difference", not an improvement.

## 11. Automated training pipeline (Later, optional)

Start this only after one model has been promoted by hand (M4.5), and only if models will be retrained regularly. The manual promotion steps become the spec for automating it.

```
workflow_dispatch (manual) ──► self-hosted GPU runner (your PC)
   train (config YAML) ─► evaluate on eval set ─► compare with eval/baseline.json
        │ pass                                       │ fail
        ▼                                            ▼
   candidate weights uploaded as artifact       job fails with metrics summary
        │
   manual approval (environment: model-release)
        ▼
   weights published as release asset + PR pinning new model version + updated baseline
```

Security: GitHub advises against self-hosted runners on public repositories, because a pull request from a fork could run code on your machine. The training workflow therefore:

- triggers only on `workflow_dispatch`, never on `pull_request`
- runs as a dedicated non-admin OS user
- requires approval for workflows from outside collaborators (repo setting)
