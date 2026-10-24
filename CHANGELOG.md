# Changelog

All notable changes are recorded here. Releases follow the plan in [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md): `v0.2.0`, `v0.5.0`, `v1.0.0`.

## Unreleased

### Added

- Development workflow, issue/PR templates, roadmap bootstrap script, lint config and pre-commit (#11)
- Shared JSON error format `{"error", "code"}` with `error_id` on 500s across gateway, detector and grouping (#12)
- Upload limits: `MAX_UPLOAD_MB`, `MAX_IMAGE_PIXELS`, `MAX_PAYLOAD_MB`; configurable `REQUEST_TIMEOUT` (#12)
- pytest suite for input validation and failure paths, with stubbed models (#12)

### Fixed

- Corrupt or non-image uploads no longer crash the detector; they return 400 (#12)
- Grouping no longer crashes on a missing `image` or malformed payload; returns 400 (#12)
- Detector and grouping no longer return internal exception text in responses (#12)
- Gateway maps detector/grouping timeouts to 504 and failures to 502 instead of a generic 500 (#12)
