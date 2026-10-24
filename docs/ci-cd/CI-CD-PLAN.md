# CI/CD Plan

Planned GitHub Actions workflows. Status: **proposed**. They are implemented in milestone M3 (CI/CD & Docker); the training workflow is in the optional *Later* milestone.

## Workflows

| Workflow file | Trigger | Runner | Jobs | Blocks merge? | Milestone |
|---|---|---|---|---|---|
| `ci.yml` | PR into `main`, `feat/**`, `fix/**`, `exp/**`, `test/**`, `chore/**`, `docs/**` | GitHub-hosted Ubuntu | ruff lint → pytest (unit + API, no weights) | Yes for `main` | M3.1 |
| `docker-build.yml` | PR into `main` touching Dockerfiles, requirements or service code | GitHub-hosted | `docker buildx` per service with GitHub Actions cache | Yes for `main` | M3.1 |
| `eval-gate.yml` | PR into `main` touching `detector_service/`, `grouping_service/`, `ml/`, model config | GitHub-hosted (CPU) | Download pinned weights → run eval on a small CI subset → compare with `eval/baseline.json` using `eval/gate.yaml` tolerances | Yes for `main` | M3.1 |
| `publish-images.yml` | Push to `main`, tags `v*` | GitHub-hosted | Build and push to `ghcr.io/firrexguptaji/retail-ai-<service>` tagged `sha-<short>` and version | — | M3.3 |
| `release.yml` | Tags `v*` | GitHub-hosted | GitHub release with CHANGELOG section, eval report and model assets with sha256 | — | M3.3 |
| `train-model.yml` | `workflow_dispatch` only (inputs: config path, run name) | **self-hosted, gpu** (your PC) | train → full eval → gate → upload candidate artifact → (approval) → publish weights + open PR pinning version | — | Later (L.1), optional |

## Model artefacts

- Weights and datasets are never committed to git.
- Released weights are GitHub release assets named `<model>-<version>.pt` with a `.sha256` file.
- Services read `DETECTOR_WEIGHTS_VERSION` / URL from config. A download script verifies the checksum before loading.
- `eval/baseline.json` records the metrics of the currently pinned weights. Promotion updates both in the same PR.

## Eval gate

`eval/gate.yaml`. Values are set after the M2 baseline exists, not before, and should be no tighter than the baseline's confidence intervals or the gate will fail on noise:

```yaml
detection:
  map50:        { max_drop: <set after baseline> }
  recall:       { max_drop: <set after baseline> }
grouping_gt_boxes:
  ari:          { max_drop: <set after baseline> }
latency_ms_p95: { max_increase_pct: <set after baseline> }
```

The CI gate uses a small, fixed subset so it finishes quickly on CPU. The full eval set runs in `train-model.yml` and locally before promotion.

## Branch protection on `main` (after M3)

- Require a pull request before merging
- Require status checks: `ci / test`, `docker-build`, `eval-gate` (when triggered)
- Require branches to be up to date
- Block force-pushes and deletion

## Self-hosted runner safety

GitHub advises against self-hosted runners on public repositories. Mitigations:

- `train-model.yml` runs only on `workflow_dispatch`; no other workflow targets `self-hosted`.
- Settings → Actions → "Require approval for all outside collaborators".
- The runner runs as a dedicated non-admin OS user with access only to its work directory and the dataset folder.
- The promotion step uses a GitHub environment `model-release` with you as required reviewer.

## Information still needed

- GPU model and VRAM, to set batch size and image size for training
- OS the runner will use (Windows native vs WSL2 / Linux)
- Disk space for SKU-110K plus training runs
