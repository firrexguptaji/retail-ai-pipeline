# Retail AI Pipeline — Roadmap

> Generated from `scripts/github/roadmap.json` by `scripts/github/bootstrap.py --render-md`. Edit the JSON, not this file. Live status lives on [project board #6](https://github.com/users/firrexguptaji/projects/6).

## Milestone order

```
M1 Foundation ─┬─► M2 Eval baseline ──┬─► M4 Model R&D ─► M5 Perf & Observability ─► M6 Docs
               └─► M3 CI/CD & Docker ─┘        │
  release:          v0.2.0 (after M2)       v0.5.0 (after M4)                       v1.0.0 (after M6)
                                                └─► Later: automated training (optional)
```

M2 and M3 can run in parallel after M1. #4 (metrics, already in progress on the board) may continue in parallel.

## Rules

- **Splitting:** An issue gets real sub-issues (each with its own branch) only when it is size L and its parts can be reviewed on their own. Everything else is one branch with a task checklist.
- **Releases:** Three releases: v0.2.0 (reliable and measured), v0.5.0 (CI/CD and improved models), v1.0.0 (fast, observable, documented).

| Milestone | Release | Issues | Goal |
|---|---|---|---|
| M1 — Foundation: Workflow, Fixes & Tests | — | 5 | Workflow in place, no request can crash a service, pytest suite covers units, APIs and failure paths. |
| M2 — Evaluation Baseline | `v0.2.0` | 2 | Labelled eval set and harness; detection, grouping and latency baseline with confidence intervals. |
| M3 — CI/CD & Docker | — | 3 | Tests, lint, Docker builds and eval gate on every PR; images and releases published automatically; weights out of git. |
| M4 — Model R&D | `v0.5.0` | 5 | Detector fine-tuned on SKU-110K, grouping experiments, best models promoted manually with model cards. |
| M5 — Performance & Observability | — | 4 | Lower latency measured against the baseline; Prometheus, Grafana and Jaeger. |
| M6 — Documentation & v1.0.0 | `v1.0.0` | 2 | Docs that match the code, walkthrough, changelog, v1.0.0. |
| Later — Automated Training Pipeline | — | 1 | Optional. Start only after one model has been promoted manually in M4. |

**Total:** 22 issues (5 existing), 18 sub-issues.

## M1 — Foundation: Workflow, Fixes & Tests

Workflow in place, no request can crash a service, pytest suite covers units, APIs and failure paths.

### `M1.1` Workflow setup and repository hygiene

*type: chore, area: docs, area: infra, priority: high, size S*

Put the workflow in writing, make GitHub enforce it, and stop committing generated files.

**Tasks** (one branch, one PR)

- [ ] Move templates into .github/ (issue templates + PR template)
- [ ] Add docs/workflow/WORKFLOW.md and a short CONTRIBUTING.md
- [ ] Run scripts/github/bootstrap.py to create labels, milestones and issues
- [ ] Protect main: PR required, no force-push (CI checks added in M3)
- [ ] Update .gitignore (data/, runs/, mlruns/, checkpoints) and add .env.example for every env var
- [ ] Add pre-commit with ruff and nbstripout

**Acceptance criteria**

- Templates render on GitHub; roadmap issues exist on board #6 with Priority and Size
- main is protected
- .env.example matches all os.getenv calls; pre-commit run --all-files passes

### `M1.2` Validate input and return safe errors

*type: bug, area: gateway, priority: high, size M*

Corrupt or oversized uploads crash the detector (cv2.imdecode returns None) and services return raw exception text.

**Tasks** (one branch, one PR)

- [ ] Gateway: MAX_UPLOAD_MB limit, content-type and decode check before calling detector
- [ ] Detector and grouping: validate payloads; 500 responses carry only an error id
- [ ] Gateway: REQUEST_TIMEOUT from env; downstream timeout → 504, other failure → 502

**Acceptance criteria**

- Corrupt/non-image upload → 400; too large → 413; no stack traces or exception text in any response
- Each case covered by a test

### `M1.3` Fix grouping edge cases and preprocessing

*type: bug, area: grouping, priority: high, size M*

One detection crashes AgglomerativeClustering; ResNet input lacks ImageNet normalisation; embeddings run one crop at a time.

**Tasks** (one branch, one PR)

- [ ] Handle 0 and 1 detections
- [ ] Add ImageNet mean/std normalisation; use torchvision weights enum instead of pretrained=True
- [ ] Batch embedding inference; move model to device once

**Acceptance criteria**

- 0, 1 and 2 detections return valid responses (tested)
- Preprocessing test checks tensor shape, dtype and value range
- Note in PR: grouping output changes; re-measured in M2

### `M1.4` Fix detector configuration

*type: bug, area: detector, priority: high, size S*

MODEL_PATH default uses Windows backslashes relative to the working directory; confidence threshold is hard-coded.

**Tasks** (one branch, one PR)

- [ ] Resolve MODEL_PATH relative to the repo; fail fast with a clear log if weights are missing
- [ ] CONF_THRESHOLD from env; Ultralytics verbose output off

**Acceptance criteria**

- Service starts from repo root on Windows, Linux and Docker
- Missing-weights path covered by a test

### `M1.5` Test framework across services

*type: test, priority: high, size L*

The project has no tests. Add a pytest suite that runs fast without model weights, plus one slow end-to-end test.

**Acceptance criteria**

- pytest runs from repo root; unit + API tests need no weights
- Failure paths tested for every endpoint
- README documents how to run tests

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `M1.5.a` | pytest setup, fake models and unit tests | YOLO and ResNet stubbed by fixtures; sliding window, filters, NMS, spatial features, clustering covered incl. image smaller than slice and no boxes |
| `M1.5.b` | API tests for gateway, detector and grouping | Flask test client; downstream calls mocked; error paths asserted |
| `M1.5.c` | End-to-end smoke test on a sample image | marked slow; real weights; asserts response schema |


## M2 — Evaluation Baseline  → releases `v0.2.0`

Labelled eval set and harness; detection, grouping and latency baseline with confidence intervals.

### `M2.1` Build labelled evaluation set

*type: feature, area: ml, priority: high, size M*

Without ground truth no accuracy claim can be checked. Start with 20 shelf images (grown to 40+ in M4.2).

**Tasks** (one branch, one PR)

- [ ] Annotation guideline: what is a product, partial products, what counts as the same SKU
- [ ] Label 20 images with boxes and SKU group IDs (Label Studio or CVAT), export in a documented format
- [ ] data/eval/ manifest with checksums; eval images never used for training

**Acceptance criteria**

- docs/evaluation/annotation-guideline.md exists
- 20 labelled images load and pass checksum verification

### `M2.2` Evaluation harness and baseline report

*type: feature, area: ml, priority: high, size L*

One command that scores detection, grouping and latency separately, with confidence intervals so small differences are not over-claimed.

**Acceptance criteria**

- Every metric reported with a 95% bootstrap confidence interval (resampling images)
- docs/evaluation/baseline.md lists workload, environment, method and results
- eval/baseline.json committed for the CI gate

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `M2.2.a` | Detection metrics | precision, recall, mAP@0.5; IoU matching unit-tested |
| `M2.2.b` | Grouping metrics on ground-truth and predicted boxes | ARI, NMI, purity on GT boxes (isolates grouping) and on predicted boxes (end-to-end); tested on toy data |
| `M2.2.c` | Latency benchmark and baseline report | fixed images, warm-up, p50/p95 per stage, peak memory; bootstrap CIs; baseline.md + baseline.json |


## M3 — CI/CD & Docker

Tests, lint, Docker builds and eval gate on every PR; images and releases published automatically; weights out of git.

### `M3.1` CI on pull requests

*type: feature, area: infra, priority: high, size M*

GitHub Actions on every PR into main.

**Tasks** (one branch, one PR)

- [ ] ci.yml: ruff + pytest (no weights)
- [ ] docker-build.yml: buildx per service with GitHub Actions cache
- [ ] eval-gate.yml: CPU eval on a small fixed subset, compared with eval/baseline.json using eval/gate.yaml tolerances
- [ ] Add these as required checks on main

**Acceptance criteria**

- A PR that breaks a test, a Dockerfile or drops metrics beyond tolerance cannot merge

### `M3.2` Improve Docker Build Time — existing #8

*type: chore, area: infra, priority: medium, size M*

Build context is the repo root with no .dockerignore, so the 1.9 GB venv is sent on every build. Images carry full OpenCV, ffmpeg and CUDA torch wheels.

**Tasks** (one branch, one PR)

- [ ] Add .dockerignore (venv, data, notebooks, outputs, logs)
- [ ] opencv-python-headless, CPU-only torch wheels, drop ffmpeg and unused open-clip
- [ ] Record context size, image size and cold/warm rebuild time before and after

**Acceptance criteria**

- Rebuild without dependency changes < 1 min (measured)
- All services start and pass the smoke test

### `M3.3` CD: images, releases and model weights out of git

*type: feature, area: infra, priority: medium, size M*

Merges to main publish images; version tags publish releases; weights become release assets.

**Tasks** (one branch, one PR)

- [ ] Push images to GHCR on merge to main (sha tag) and on v* tags
- [ ] release.yml on v* tags: CHANGELOG section + eval report
- [ ] Weights as release assets with sha256; download script verifies before load; remove models/yolov8n.pt from git

**Acceptance criteria**

- docker-compose can run from published images
- Fresh clone downloads and verifies weights with one command


## M4 — Model R&D  → releases `v0.5.0`

Detector fine-tuned on SKU-110K, grouping experiments, best models promoted manually with model cards.

### `M4.1` Restructure notebooks around shared code

*type: feature, area: ml, priority: high, size M*

The current notebook duplicates service code. Notebooks should import the same preprocessing and metrics as the services so results transfer.

**Tasks** (one branch, one PR)

- [ ] Create ml/ package (preprocessing, metrics, visualisation) used by services, harness and notebooks
- [ ] Notebook conventions: numbered, parameter cell, fixed seeds, outputs stripped; experiment report template
- [ ] 00_data_exploration: SKU-110K and own images (resolution, objects per image, box size/aspect)
- [ ] 01_baseline_reproduce: reproduces eval/baseline.json

**Acceptance criteria**

- Both notebooks run top to bottom
- 01 matches baseline.json within its confidence intervals

### `M4.2` Grow evaluation set to 40+ images

*type: feature, area: ml, priority: high, size M*

20 images give wide confidence intervals. Grow the set before model decisions are made.

**Tasks** (one branch, one PR)

- [ ] Label 20+ more images following the guideline, covering lighting, angle and shelf types
- [ ] Re-run baseline; update baseline.md/json and gate tolerances

**Acceptance criteria**

- 40+ labelled images
- Baseline CIs reported narrower than with 20 images

### `M4.3` Fine-tune YOLOv8 on SKU-110K

*type: experiment, area: detector, area: ml, priority: high, size L, depends on M2.2, M4.1, M4.2*

Replace the COCO-pretrained detector with one trained on dense retail shelves, on the local GPU, judged on our own eval set.

**Acceptance criteria**

- docs/experiments/EXP-001-detector-finetune.md with results table, CIs and a decision
- Runs logged with params, metrics and weights hash

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `M4.3.a` | Prepare SKU-110K: licence, disk space, YOLO format | licence terms recorded in the report; disk space checked before download; scripted download + conversion |
| `M4.3.b` | Train yolov8n vs yolov8s, image size sweep; full-image vs sliding window | accuracy and p50/p95 latency for each variant |
| `M4.3.c` | Sweep confidence, NMS (confidence-ranked) and area filters; write EXP-001 | chosen settings justified by eval metrics with CIs |

### `M4.4` Grouping experiments

*type: experiment, area: grouping, area: ml, priority: high, size L, depends on M2.2, M4.1, M4.2*

Measure each grouping choice on ground-truth boxes, one variable at a time.

**Acceptance criteria**

- docs/experiments/EXP-002-grouping.md with results table, CIs and a decision

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `M4.4.a` | Crop preprocessing and embedding backbones | pad vs stretch, no JPEG round-trip; ResNet18, ResNet50, CLIP ViT-B/32, DINOv2-S: accuracy and per-crop latency |
| `M4.4.b` | Spatial features: measure contribution, weight sweep | visual vs spatial norm ratio reported |
| `M4.4.c` | Clustering: threshold/linkage sweep, cosine, HDBSCAN; write EXP-002 | threshold sensitivity plot |

### `M4.5` Promote best models (manual)

*type: feature, area: ml, priority: medium, size M, depends on M4.3, M4.4*

Ship the winners through config, by hand, once. This manual run is the prerequisite for automating training later.

**Tasks** (one branch, one PR)

- [ ] Publish chosen weights as a release asset; pin version in config
- [ ] Model cards in docs/models/ (data, training, metrics with CIs, limits, licence)
- [ ] Update eval/baseline.json; before/after table in PR

**Acceptance criteria**

- All tests and eval gate pass with the new models
- Promotion steps written down (input for the Later milestone)


## M5 — Performance & Observability

Lower latency measured against the baseline; Prometheus, Grafana and Jaeger.

### `M5.1` Optimize Detector Performance — existing #7

*type: feature, area: detector, priority: medium, size L, depends on M4.5*

Reduce end-to-end latency without losing accuracy. Starts after M4 because the fine-tuned model may remove the need for sliding windows.

**Acceptance criteria**

- Latency target set from the baseline on a fixed workload and machine
- Accuracy within eval gate tolerance
- Before/after report in docs/evaluation/

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `M5.1.a` | Batch sliding-window and embedding inference | one model call per batch (or sliding window removed if M4 showed full-image is enough) |
| `M5.1.b` | Cut inter-service payload size | payload bytes per request before/after |
| `M5.1.c` | Tune gunicorn workers/threads/timeouts and benchmark | memory per worker documented; p50/p95 and accuracy table vs baseline |

### `M5.2` Add Prometheus metrics to all services — existing #4

*type: feature, area: observability, priority: medium, size M*

Expose /metrics in gateway, detector and grouping (already In progress on the board).

**Tasks** (one branch, one PR)

- [ ] Shared metrics helper in common/
- [ ] Request count, error count, latency histogram per endpoint in all three services
- [ ] Model metrics: inference time histogram, detections per image, groups per image

**Acceptance criteria**

- /metrics on all three services, tested via test client

### `M5.3` Add Prometheus + Grafana — existing #6

*type: feature, area: observability, area: infra, priority: medium, size M, depends on M5.2*

Prometheus and Grafana in docker-compose with a provisioned dashboard.

**Tasks** (one branch, one PR)

- [ ] Prometheus service and scrape config
- [ ] Grafana datasource and dashboard JSON provisioned from the repo

**Acceptance criteria**

- Prometheus UI on 9090 with all targets UP
- Dashboard loads with no manual steps

### `M5.4` Add distributed tracing with OpenTelemetry + Jaeger — existing #5

*type: feature, area: observability, priority: medium, size M*

Trace gateway → detector → grouping with per-stage spans.

**Tasks** (one branch, one PR)

- [ ] OpenTelemetry SDK with Flask and requests instrumentation
- [ ] Jaeger in docker-compose
- [ ] Custom spans: decode, inference, NMS, embedding, clustering

**Acceptance criteria**

- One trace per request in Jaeger with stage timings


## M6 — Documentation & v1.0.0  → releases `v1.0.0`

Docs that match the code, walkthrough, changelog, v1.0.0.

### `M6.1` Complete documentation set

*type: chore, area: docs, priority: medium, size M*

Docs that match the code.

**Tasks** (one branch, one PR)

- [ ] README: accurate structure, quickstart, results table
- [ ] docs/architecture.md with diagram and data flow
- [ ] docs/deployment.md: env vars, ports, weights, troubleshooting
- [ ] Short ADRs for the main decisions, each citing its experiment report

**Acceptance criteria**

- Every command in README and deployment guide verified on a fresh clone

### `M6.2` v1.0.0 release

*type: chore, area: docs, area: infra, priority: medium, size S*

Changelog, walkthrough with screenshots, tagged release.

**Tasks** (one branch, one PR)

- [ ] CHANGELOG for v0.2.0, v0.5.0, v1.0.0
- [ ] Walkthrough with UI, Grafana and Jaeger screenshots
- [ ] Tag v1.0.0; release published by CD

**Acceptance criteria**

- Release has images, weights and eval report attached


## Later — Automated Training Pipeline

Optional. Start only after one model has been promoted manually in M4.

### `L.1` Automated model training pipeline (optional)

*type: feature, area: ml, area: infra, priority: low, size L, depends on M4.5, M3.3*

Automate what M4.5 did by hand: train → evaluate → gate → promote, on the local GPU via a self-hosted runner. Only worth doing if models will be retrained regularly.

**Acceptance criteria**

- Runner triggered only by workflow_dispatch; never by pull_request
- Promotion requires manual approval (environment: model-release)

**Sub-issues**

| Key | Sub-issue | Done when |
|---|---|---|
| `L.1.a` | Training script with YAML config | same result as notebook for same config and seed |
| `L.1.b` | Hardened self-hosted GPU runner | dedicated non-admin user; outside-collaborator workflows need approval |
| `L.1.c` | train-model workflow with gate and approved promotion | gate result in workflow summary; promotion opens PR pinning new weights |

