# AGENTS.md

Python 3.12+ manga/image translation pipeline (Torch 2.14.1, FastAPI 0.143 server, React Router 7 UI in `front/`) run from the repo root with `python -m manga_translator`; model weights download into `./models` on first use.

## Commands

- Install: `python -m venv venv`, then `pip install -r requirements.txt -r requirements-dev.txt` (CI installs both files).
- Build: `make build-image` (Docker image `manga-image-translator`); frontend bundle: `npm --prefix front run build`.
- Test all: `pytest test`
- Test one: `pytest test/test_render.py::test_default_renderer`
- Test a translator: `pytest test/test_translation_manual.py --translator sugoi --target-lang ENG`
- Lint: `pylint manga_translator`; UI typecheck: `npm --prefix front run typecheck`. CI runs pylint over tracked `.py` files with `continue-on-error`, so only `pytest` gates.

## Boundaries

### Always

- Finish with `pytest test`; `.github/workflows/ci.yml` runs it on Python 3.12 and 3.13 and is the only failing gate.
- Add or update tests under `test/` for behaviour you change; render and textline tests run offline, `test_translation*.py` calls live translator APIs.
- Keep a translator name in sync across `manga_translator/config.py`, `manga_translator/translators/__init__.py`, `manga_translator/translators/keys.py`, `MangaStudio_Data/app/core/constants.py`, `examples/gpt_config-example.yaml`, and the README translator tables.

### Ask first

- Adding or bumping a pin in `requirements*.txt`; CI and the Dockerfile install these exact files.
- Changing enums or defaults in `manga_translator/config.py`; they are the user-facing JSON config surface printed by `python -m manga_translator config-help`.
- Editing `.github/workflows/release.yml` or any Docker Hub publishing setting.

### Never

- Commit or print `.env` contents; `manga_translator/translators/keys.py` loads API keys from it and `.gitignore` excludes it.
- Commit `models/`, `result/`, `test/testdata/`, or `front/build/`; these are runtime downloads and build outputs.
- Edit `*.pt`, `*.onnx`, or `*.ckpt` weights.

## Standalone HTTP server

- `standalone/` is the fork's trimmed "POST image → get typeset image" server; `standalone/API.md` is the complete client contract (endpoints, profiles incl. `auto`, the `X-MIMG-Profile` header, errors, latency). Read it before calling or changing that server; keep it updated when its behavior changes.

## Worker model choice

- Delegate to a local-model worker before a remote paid one: always for sensitive information or content likely against remote-provider terms of service, and whenever the task is within local capability — minimize remote token usage when possible.
- Local workers run sequentially, so the main agent blocks on them; when a local and a remote worker are needed at the same time, launch the remote worker first so its work overlaps the local run.

## Exploratory tasks: investigate efficiently, then act
Applies to: design, debugging, refactoring, planning, or working in unfamiliar code.

Context is limited. Reading too much triggers compaction and loses the work. Explore like a budget, not a survey.

**Read cheaply first**
- Use search (grep, symbol lookup, file listings) to locate relevant code before opening any file.
- Read the specific function, class, or line range you need, not whole files.
- Don't open a file just to confirm something you can reasonably infer.
- When locating the answer would fan out across 4+ files or wade through thousands of lines, delegate the hunt to the `scout` worker instead of reading serially — that is what it exists for.

**Stop when you can act**
- Stop exploring as soon as you can name the next concrete change and why it's correct.
- Don't aim for complete understanding of the codebase, only of the code this change touches.
- If a question can be settled by trying the change and running tests/build, do that instead of reading more.

**Work in small steps**
- Split large tasks into atomic pieces. Finish and verify one before exploring for the next.
- Explore only what the current step needs, not what later steps might need.

**Protect findings from compaction**
- After a meaningful discovery, record it briefly (key file paths, function names, decisions, open questions) in a notes file such as `.agent-notes.md`, or in your plan.
- If the context was compacted, re-read the notes file instead of re-exploring.
- Keep notes short: conclusions, not file contents.

**If exploration is dragging on**
- If you've read many files without a clear next action, stop, summarize what you know, and either state your best-supported plan, ask me a specific question, or — if the blocker is a reasoning knot rather than missing information — hand the knot to the `analyst` worker.

## Research discipline

### source-criticism

- Ranking pages, listicles, and forum threads are leads, not evidence. Extract their claims, then check each against at least two independent sources.
- Prefer primary sources for rules, APIs, and version facts: vendor docs, standards bodies, the repository itself, official changelogs. Blogs and commentary rank below them.
- Ask who benefits from the claim being true before accepting it, especially for "best X" comparisons written by the tool's own vendor.

### cross-verification

- Every number must come from a source opened in this session. Quote URL and access date; add the effective year when the figure changes over time.
- Run one deliberate disconfirming pass: complaints, deprecations, recalls, changed or discontinued status, contrary expert findings.
- Show disagreements instead of resolving them quietly, and say which side is better supported.

### citations

- Cite inline next to the claim, not only in a closing list. Separate what is confirmed from what is inferred from what is unverified.
- Label anything you could not check live as UNVERIFIED. Never promote an unsourced number into a table.
- Record confidence per finding: High (multiple independent primary sources), Medium (one primary or several secondary), Low (single unconfirmed source).

### synthesis

- State the answer's scope and staleness risk before recommending. Prices, versions, and rates expire within days or weeks.
- Name the deciding factor explicitly. If the recommendation flips on one fact, say which fact and where to verify it.

For coding discipline, read `.agents/rules/code.md`.
