# Contributing

The full process is in [docs/workflow/WORKFLOW.md](docs/workflow/WORKFLOW.md). The short version:

1. **Pick an issue** from [project board #6](https://github.com/users/firrexguptaji/projects/6) that is in *Ready*. Move it to *In progress*.
2. **Branch from `main`:** `<type>/<issue#>-<slug>`, for example `fix/12-input-validation`.
   Only size-L issues with sub-issues use sub-branches: `sub/<issue#>/<sub#>-<slug>`.
3. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/) and the issue number: `fix(gateway): reject corrupt uploads (#12)`.
4. **Open a PR** using the template. `Closes #<issue>` for PRs into `main`.
5. **Done** means: acceptance criteria met, tests pass, docs and CHANGELOG updated, merged.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Mac/Linux)
pip install -r requirements/requirements-dev.txt
pip install pre-commit && pre-commit install
copy .env.example .env           # cp on Mac/Linux
```

## Model and data changes

- Never commit datasets or model weights.
- Changes to detection or grouping need a before/after metrics table from the eval harness (from M2 onward).
- Experiments follow [docs/experiments/TEMPLATE.md](docs/experiments/TEMPLATE.md).

## Roadmap

Edit `scripts/github/roadmap.json`, then regenerate the docs:

```bash
python scripts/github/bootstrap.py --render-md docs/roadmap/ROADMAP.md
```
