# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A staged series of AI agents, each building on the last: starting from a bare
read-act loop (`agent-001-echo-loop`) and progressing toward agents with LLM
calls, tools, and evals. Each stage is paired with a blog post explaining it,
published via GitHub Pages from `docs/`.

## Repo structure

- `agents/agent-NNN-<slug>/` -- one self-contained folder per stage. Each
  folder holds that stage's `agent.py`, its `test_agent.py`, and a small
  `conftest.py` (sys.path insertion + evicting any cached `agent` module).
  Folders have no `__init__.py`. Since every stage's `test_agent.py` shares
  the same basename and does `from agent import run` without
  package-qualifying imports, pytest's default import mode can't
  disambiguate them once more than one stage exists in a single test run --
  `pyproject.toml` sets `--import-mode=importlib` and each stage's
  `conftest.py` makes the unqualified import resolve to that stage's own
  `agent.py`. Don't add `__init__.py` to these folders; do add a
  `conftest.py` (copy an existing stage's) to any new one.
- `docs/` -- the GitHub Pages site (Jekyll, `jekyll-theme-minimal`).
  `docs/_posts/` holds one post per stage, named
  `YYYY-MM-DD-agent-NNN-<slug>.md` with Jekyll front matter
  (`layout: post`, `title`, `date`). `docs/index.md` lists all posts
  automatically via `site.posts` -- no manual index maintenance needed.
- `pyproject.toml` / `uv.lock` -- repo-wide dependencies, managed with `uv`.
  Stage-specific dependencies (e.g. an LLM SDK for later agents) should be
  added with `uv add <package>` as they're introduced, since stages are
  meant to be run from a single shared environment.

## Commands

- Install/sync dependencies: `uv sync`
- Run all tests: `uv run pytest agents/`
- Run a single stage's tests: `uv run pytest agents/agent-001-echo-loop/`
- Run a single stage's agent directly: `uv run python agents/agent-001-echo-loop/agent.py`

## Conventions

- Python, adapted naming: `PascalCase` for classes, `snake_case` for
  functions/variables/files, `ALL_CAPS` for constants. (Not camelCase --
  that's a JS convention and doesn't apply here.)
- Each agent stage takes `input_stream`/`output_stream` (or equivalent)
  as parameters rather than hardcoding `sys.stdin`/`sys.stdout`, so it can be
  tested with `io.StringIO` instead of spawning a subprocess. Keep this
  pattern for new stages.
- Every stage's blog post explains *why* that stage's mechanism works the
  way it does, not just what the code does -- write posts assuming the
  reader has the code open alongside the post.
- Each stage's post ends with a "What's next" section linking forward to
  the next stage's post. Link between posts with
  `[Text]({{ site.baseurl }}{% post_url YYYY-MM-DD-agent-NNN-<slug> %})` --
  never a raw `.md` filename (it 404s) and never a hardcoded `/baseurl/`
  prefix. `{% post_url %}` resolves the permalink and errors the build on a
  bad slug; `{{ site.baseurl }}` supplies the GitHub Pages path prefix.
- Tests are written only where there's real logic to verify (eg. loop termination conditions); trivial stages don't need padding for coverage's sake.
