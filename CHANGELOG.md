# Changelog

All notable changes are recorded here. Releases follow the plan in [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md): `v0.2.0`, `v0.5.0`, `v1.0.0`.

## Unreleased

### Added

- Development workflow, issue/PR templates, roadmap bootstrap script, lint config and pre-commit (#11)
- Shared JSON error format `{"error", "code"}` with `error_id` on 500s across gateway, detector and grouping (#12)
- Upload limits: `MAX_UPLOAD_MB`, `MAX_IMAGE_PIXELS`, `MAX_PAYLOAD_MB`; configurable `REQUEST_TIMEOUT` (#12)
- pytest suite for input validation and failure paths, with stubbed models (#12)

### Fixed

- Detector fails fast with a clear error when the weights file is missing, instead of resolving `..\models\yolov8n.pt` outside the repo, where Ultralytics may load or download a different file (#14)
- Child loggers no longer print every line twice (#49)
- Grouping no longer crashes when an image has exactly one detection (#13)
- ResNet18 input now uses ImageNet mean/std normalisation; group assignments change as a result and `DISTANCE_THRESHOLD` will be re-tuned against the M2 baseline (#13)
- Replaced deprecated `pretrained=True` with `ResNet18_Weights.IMAGENET1K_V1` (same weights) (#13)
- Corrupt or non-image uploads no longer crash the detector; they return 400 (#12)
- Grouping no longer crashes on a missing `image` or malformed payload; returns 400 (#12)
- Detector and grouping no longer return internal exception text in responses (#12)
- Gateway maps detector/grouping timeouts to 504 and failures to 502 instead of a generic 500 (#12)

### Changed

- `MODEL_PATH` resolves relative paths from the repo root (not the working directory) and accepts `\` or `/`; default is `models/yolov8n.pt` (#14)
- Detector confidence threshold is configurable via `CONF_THRESHOLD` (default unchanged at 0.03); per-window Ultralytics console output is off (#14)
- Grouping embeds crops in batches (`EMBED_BATCH_SIZE`, default 32) and moves the model to the device once at startup; `DEVICE=cuda` falls back to CPU when CUDA is unavailable (#13)
