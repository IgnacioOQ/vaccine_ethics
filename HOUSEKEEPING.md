---
status: active
type: workflow
description: Periodic repo audit for the vaccine_ethics research codebase — engine tests, notebook Restart-and-Run-All, figure-traceability and dependency health checks, with an append-only audit trail.
label: [agent, template]
injection: excluded
volatility: evolving
scope: project-specific
last_checked: '2026-06-23'
---

# Housekeeping Workflow

Per-repository audit checklist for `vaccine_ethics`, instantiated from the
knowledge base's `HOUSEKEEPING_TEMPLATE.md`. Run it periodically or after any
structural change. Because this is a research repo (the unit of correctness is
the figure, not the feature), the "tests" phase is light: engine smoke tests
plus a notebook Restart-and-Run-All sweep.

**Execution model:** sequential — each phase has an exit criterion.

**Prerequisites:**

- A Python env with `requirements.txt` installed (plus `pytest` for Phase 3).
- `worklog.jsonl` and `TODO_WORKFLOW.md` at the repo root.

## JSONL log

The archive of prior reports lives at `housekeeping_log.jsonl` at the repo root
(append-only, one JSON record per line, oldest-first). The companion
`worklog.jsonl` (session history) is documented in `TODO_WORKFLOW.md`. Record
schema (`schema_version: 1`): `entry_id` (`YYYY-MM-DD`), `date`, `trigger`,
`metrics` (free-form dict), `body_markdown`. Bootstrap with
`touch housekeeping_log.jsonl`.

## Phase 1 — Context load

1. Read this file's prior `## Latest Report` for the baseline.
2. Confirm the toolchain: `pip install -r requirements.txt` (+ `pytest`).

**Exit:** baseline read; commands known.

## Phase 2 — Static quality

Research code carries no lint/format merge gate. Spot-check only:

1. No stray Colab/Drive paths in active notebooks:
   `grep -rl "/content/drive" 0*/ ` should return nothing.
2. No build artifacts tracked: `git ls-files | grep -E "__pycache__|\.DS_Store"`
   should return nothing.

**Exit:** both greps empty.

## Phase 3 — Tests

1. Engine smoke tests:

   ```bash
   pytest -q
   ```

2. Notebook Restart-and-Run-All (the honest equivalent of an integration
   suite). Long Bayesian-optimization notebooks may be skipped or smoke-toggled;
   note any skips in the report.

   ```bash
   for nb in $(find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" \
                       -not -path "./.conda/*" -not -path "./archive/*"); do
     jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=900 "$nb"
   done
   ```

**Exit:** `pytest` green; notebooks execute clean (or skips are recorded).

## Phase 4 — Repository health

1. **Figure traceability** — every figure in `results/` is listed in
   `results/MANIFEST.md`, and unattributed entries are resolved or removed.
2. **Dependency drift** — `pip list --outdated`; refresh `requirements.txt` if a
   pin is materially stale.
3. **Docs freshness** — `README.md` strand map and run commands still match the
   tree; resolved `TODO_WORKFLOW.md` items cleaned up.

**Exit:** no surprising drift; anything actionable filed in `TODO_WORKFLOW.md`.

## Phase 5 — Report & close

1. Replace `## Latest Report` below with this run's results.
2. Append the prior report to `housekeeping_log.jsonl`.
3. File follow-ups in `TODO_WORKFLOW.md`.
4. Bump `last_checked` in this file's frontmatter.

## Quick reference

```text
[ ] Phase 1: baseline read, toolchain confirmed
[ ] Phase 2: no Colab paths, no tracked build artifacts
[ ] Phase 3: pytest green; notebooks run (or skips recorded)
[ ] Phase 4: figures traceable; deps + docs current
[ ] Phase 5: report appended; prior archived; last_checked bumped
```

## Latest Report

**Date:** 2026-06-23
**Trigger:** Periodic audit run (`/HOUSEKEEPING.md`).

```yaml
format:        n/a
lint:          n/a
types:         n/a
tests:         { engine_smoke: pass (6 passed), notebooks: not_run }
integration:   n/a
dependencies:  ok
dead_code:     ok
build:         n/a
docs:          ok (one MANIFEST traceability gap fixed)
```

### Notable

`pytest` now green — **6 passed** in the `.conda` env (`pytest`, `nbconvert`,
`ipykernel` were missing and were installed). Phase 2.2 clean: no tracked build
artifacts. Phase 4 dependency/doc checks clean: core pins not flagged outdated,
README strand map and run commands still match the tree. Fixed a figure-
traceability gap — added the two present-but-unlisted seeding CSVs
(`bo_results_fixed_params_seeding.csv`, `vax_hurts_region_seeding.csv`) to
`results/MANIFEST.md`; their presence partially advances `todo.regenerate_results`.

The notebook Restart-and-Run-All sweep was **skipped** (sanctioned by Phase 3):
the two heavy notebooks (`Run_Simulations_Final.ipynb`,
`Precautionary_Principle_Analysis.ipynb`) carry `/content/drive` Colab paths and
would fail unattended; the remaining validation notebooks were not executed.

### Outstanding

- **Phase 2.1 fails:** `grep -rl "/content/drive" 0*/` still returns
  `01_optimization/Run_Simulations_Final.ipynb` and
  `02_precautionary/Precautionary_Principle_Analysis.ipynb`. Strip the Colab/Drive
  paths so the notebooks are portable and runnable in the sweep.
- Execute the optimization/precautionary notebooks end-to-end (`todo.regenerate_results`).
- Resolve the unattributed PNGs (`example_plot.png`, `output.png`) and the
  misspelled duplicate `precautionaty_plot.png` in `results/MANIFEST.md`.
