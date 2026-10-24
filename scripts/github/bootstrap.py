#!/usr/bin/env python3
"""Create labels, milestones, issues and sub-issues from roadmap.json.

Usage (from repo root):
    python scripts/github/bootstrap.py                  # dry run: print what would happen
    python scripts/github/bootstrap.py --render-md docs/roadmap/ROADMAP.md
    python scripts/github/bootstrap.py --apply          # create on GitHub
    python scripts/github/bootstrap.py --apply --add-to-project
    python scripts/github/bootstrap.py --apply --rewrite-existing   # also rewrite bodies of #4-#8

Requirements: GitHub CLI (`gh`) authenticated with repo scope.
`--add-to-project` additionally needs: gh auth refresh -s project

Idempotent: created issues are recorded in scripts/github/.bootstrap-state.json
and matched by title, so re-running never duplicates anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROADMAP = HERE / "roadmap.json"
STATE = HERE / ".bootstrap-state.json"

BRANCH_TYPE = {
    "type: bug": "fix",
    "type: experiment": "exp",
    "type: test": "test",
    "type: chore": "chore",
    "type: feature": "feat",
}

DOD = [
    "Acceptance criteria met",
    "Tests added/updated and passing in CI",
    "Docs updated (README / docs/ / experiment report as relevant)",
    "CHANGELOG `Unreleased` section updated",
    "Merged through PR with review checklist complete",
]


# --------------------------------------------------------------------------- helpers
def load() -> dict:
    return json.loads(ROADMAP.read_text(encoding="utf-8"))


def gh(args: list[str], payload: dict | None = None, check: bool = True) -> str:
    cmd = ["gh", *args]
    if payload is not None:
        cmd += ["--input", "-"]
    res = subprocess.run(
        cmd,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stderr.strip()}")
    return res.stdout


def branch_type(labels: list[str]) -> str:
    for lbl in labels:
        if lbl in BRANCH_TYPE:
            return BRANCH_TYPE[lbl]
    return "feat"


def title_for(item: dict) -> str:
    return item["title"] if item.get("existing") else f"[{item['key']}] {item['title']}"


def checklist(items: list[str]) -> str:
    return "\n".join(f"- [ ] {i}" for i in items)


def parent_body(issue: dict, keys_to_titles: dict) -> str:
    parts = [
        f"**Roadmap key:** `{issue['key']}` · **Size:** {issue.get('size', '?')} · "
        f"**Branch:** `{branch_type(issue['labels'])}/<this-issue-number>-<slug>`",
        "## Summary\n" + issue["summary"],
    ]
    if issue.get("tasks"):
        parts.append("## Tasks\nOne branch, one PR into `main`.\n\n" + checklist(issue["tasks"]))
    if issue.get("sub_issues"):
        parts.append(
            "## Sub-issues\nEach sub-issue gets its own branch off this issue's branch "
            "(`sub/<this-number>/<sub-number>-<slug>`) and merges back here by PR.\n\n"
            + "\n".join(f"- `{s['key']}` {s['title']}" for s in issue["sub_issues"])
        )
    parts.append("## Acceptance criteria\n" + checklist(issue["acceptance"]))
    if issue.get("depends_on"):
        deps = ", ".join(f"`{d}` {keys_to_titles.get(d, '')}" for d in issue["depends_on"])
        parts.append("## Depends on\n" + deps)
    parts.append("## Definition of Done\n" + checklist(DOD))
    return "\n\n".join(parts)


def sub_body(sub: dict, parent: dict) -> str:
    return "\n\n".join(
        [
            f"**Roadmap key:** `{sub['key']}` · **Parent:** `{parent['key']}` {parent['title']}\n"
            f"**Branch:** `sub/<parent-number>/<this-number>-<slug>` → PR into the parent issue branch",
            "## Acceptance criteria\n" + checklist(sub["acceptance"]),
            "## Done when\n" + checklist(
                ["Acceptance criteria met", "Tests added/updated", "PR merged into parent issue branch",
                 "Sub-issue closed manually (auto-close only works for PRs into main)"]
            ),
        ]
    )


# --------------------------------------------------------------------------- markdown
def render_md(rm: dict) -> str:
    out = [
        "# Retail AI Pipeline — Roadmap",
        "",
        "> Generated from `scripts/github/roadmap.json` by `scripts/github/bootstrap.py --render-md`. "
        "Edit the JSON, not this file. Live status lives on "
        f"[project board #{rm['project']['number']}](https://github.com/users/{rm['project']['owner']}/projects/{rm['project']['number']}).",
        "",
        "## Milestone order",
        "",
        "```",
        "M1 Foundation ─┬─► M2 Eval baseline ──┬─► M4 Model R&D ─► M5 Perf & Observability ─► M6 Docs",
        "               └─► M3 CI/CD & Docker ─┘        │",
        "  release:          v0.2.0 (after M2)       v0.5.0 (after M4)                       v1.0.0 (after M6)",
        "                                                └─► Later: automated training (optional)",
        "```",
        "",
        "M2 and M3 can run in parallel after M1. #4 (metrics, already in progress on the board) may continue in parallel.",
        "",
        "## Rules",
        "",
        f"- **Splitting:** {rm['rules']['split']}",
        f"- **Releases:** {rm['rules']['releases']}",
        "",
        "| Milestone | Release | Issues | Goal |",
        "|---|---|---|---|",
    ]
    for m in rm["milestones"]:
        n = len([x for x in rm["issues"] if x["milestone"] == m["key"]])
        out.append(f"| {m['title']} | {('`' + m['release'] + '`') if m.get('release') else '—'} | {n} | {m['description']} |")
    n_sub = sum(len(i.get("sub_issues", [])) for i in rm["issues"])
    out += ["", f"**Total:** {len(rm['issues'])} issues ({sum(1 for i in rm['issues'] if i.get('existing'))} existing), {n_sub} sub-issues."]
    for m in rm["milestones"]:
        rel = f"  → releases `{m['release']}`" if m.get("release") else ""
        out += ["", f"## {m['title']}{rel}", "", m["description"], ""]
        for i in [x for x in rm["issues"] if x["milestone"] == m["key"]]:
            ex = f" — existing #{i['existing']}" if i.get("existing") else ""
            out.append(f"### `{i['key']}` {i['title']}{ex}")
            out.append("")
            out.append(
                f"*{', '.join(i['labels'])}, priority: {i['priority']}, size {i.get('size', '?')}"
                + (f", depends on {', '.join(i['depends_on'])}" if i.get("depends_on") else "")
                + "*"
            )
            out += ["", i["summary"], ""]
            if i.get("tasks"):
                out += ["**Tasks** (one branch, one PR)", ""] + [f"- [ ] {t}" for t in i["tasks"]] + [""]
            out += ["**Acceptance criteria**", ""]
            out += [f"- {a}" for a in i["acceptance"]]
            if i.get("sub_issues"):
                out += ["", "**Sub-issues**", "", "| Key | Sub-issue | Done when |", "|---|---|---|"]
                for s in i["sub_issues"]:
                    out.append(f"| `{s['key']}` | {s['title']} | {'; '.join(s['acceptance'])} |")
            out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- apply
def apply(rm: dict, dry: bool, add_to_project: bool, rewrite_existing: bool) -> None:
    repo = rm["repo"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    titles = {i["key"]: i["title"] for i in rm["issues"]}

    def say(msg: str) -> None:
        print(("[dry-run] " if dry else "") + msg)

    # Labels
    for lbl in rm["labels"]:
        say(f"label  {lbl['name']}")
        if not dry:
            gh(["label", "create", lbl["name"], "--repo", repo, "--color", lbl["color"],
                "--description", lbl["description"], "--force"])

    # Milestones
    existing_ms = {}
    if not dry:
        for line in gh(["api", "--paginate", f"repos/{repo}/milestones?state=all&per_page=100",
                        "--jq", ".[] | {title, number}"]).splitlines():
            m = json.loads(line)
            existing_ms[m["title"]] = m["number"]
    ms_num = {}
    for m in rm["milestones"]:
        if m["title"] in existing_ms:
            ms_num[m["key"]] = existing_ms[m["title"]]
            say(f"milestone exists  {m['title']}")
            continue
        say(f"milestone create  {m['title']}")
        if not dry:
            res = json.loads(gh(["api", f"repos/{repo}/milestones", "--method", "POST"],
                                {"title": m["title"], "description": m["description"]
                                 + (f" Ends with release {m['release']}." if m.get("release") else "")}))
            ms_num[m["key"]] = res["number"]

    # Existing issues by title (for idempotency)
    by_title = {}
    if not dry:
        for line in gh(["api", "--paginate", f"repos/{repo}/issues?state=all&per_page=100",
                        "--jq", ".[] | select(.pull_request == null) | {title, number, id}"]).splitlines():
            it = json.loads(line)
            by_title[it["title"]] = it

    def ensure_issue(key: str, title: str, body: str, labels: list[str], milestone_key: str,
                     existing: int | None) -> tuple[int, int] | None:
        """Return (number, id)."""
        if dry:
            say(f"issue  {'update #' + str(existing) if existing else 'create'}  {title}")
            return None
        ms = ms_num[milestone_key]
        if existing:
            cur = json.loads(gh(["api", f"repos/{repo}/issues/{existing}"]))
            patch = {"milestone": ms,
                     "labels": sorted({*(lbl["name"] for lbl in cur["labels"]), *labels})}
            if rewrite_existing and "Roadmap key:" not in (cur.get("body") or ""):
                patch["body"] = (body + "\n\n---\n<details><summary>Original description</summary>\n\n"
                                 + (cur.get("body") or "") + "\n</details>")
            gh(["api", f"repos/{repo}/issues/{existing}", "--method", "PATCH"], patch)
            say(f"issue  updated #{existing}  {title}")
            res = (existing, cur["id"])
        elif key in state:
            res = tuple(state[key])
            say(f"issue  exists #{res[0]}  {title}")
        elif title in by_title:
            res = (by_title[title]["number"], by_title[title]["id"])
            say(f"issue  exists #{res[0]}  {title}")
        else:
            created = json.loads(gh(["api", f"repos/{repo}/issues", "--method", "POST"],
                                    {"title": title, "body": body, "labels": labels, "milestone": ms}))
            res = (created["number"], created["id"])
            say(f"issue  created #{res[0]}  {title}")
        state[key] = list(res)
        STATE.write_text(json.dumps(state, indent=2))
        if add_to_project:
            p = rm["project"]
            gh(["project", "item-add", str(p["number"]), "--owner", p["owner"],
                "--url", f"https://github.com/{repo}/issues/{res[0]}"], check=False)
        return res

    for issue in rm["issues"]:
        labels = [*issue["labels"], f"priority: {issue['priority']}"]
        parent = ensure_issue(issue["key"], title_for(issue), parent_body(issue, titles),
                              labels, issue["milestone"], issue.get("existing"))
        for sub in issue.get("sub_issues", []):
            sub_labels = [*issue["labels"], f"priority: {issue['priority']}"]
            child = ensure_issue(sub["key"], f"[{sub['key']}] {sub['title']}", sub_body(sub, issue),
                                 sub_labels, issue["milestone"], None)
            say(f"  link sub-issue {sub['key']} → {issue['key']}")
            if not dry:
                # 422 when already linked: ignore
                gh(["api", f"repos/{repo}/issues/{parent[0]}/sub_issues", "--method", "POST",
                    "-F", f"sub_issue_id={child[1]}"], check=False)

    print("\nDone." if not dry else "\nDry run complete. Re-run with --apply to create these on GitHub.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="create on GitHub (default is dry run)")
    ap.add_argument("--add-to-project", action="store_true", help="also add every issue to the project board")
    ap.add_argument("--rewrite-existing", action="store_true",
                    help="rewrite bodies of existing issues (#4-#8); original text kept in a collapsed block")
    ap.add_argument("--render-md", metavar="PATH", help="write the roadmap as Markdown and exit")
    a = ap.parse_args()
    rm = load()
    if a.render_md:
        Path(a.render_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.render_md).write_text(render_md(rm), encoding="utf-8")
        print(f"Wrote {a.render_md}")
        return
    try:
        apply(rm, dry=not a.apply, add_to_project=a.add_to_project, rewrite_existing=a.rewrite_existing)
    except RuntimeError as e:
        sys.exit(f"gh call failed:\n{e}")


if __name__ == "__main__":
    main()
