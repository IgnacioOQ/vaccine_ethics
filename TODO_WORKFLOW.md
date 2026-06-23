---
status: active
type: plan
id: vaccine_ethics.todo_workflow
description: Cross-session task backlog for the vaccine_ethics repo; each task is self-contained and pickup-ready by a coding agent.
label: [planning, agent]
injection: excluded
volatility: evolving
owner: agent
last_checked: '2026-06-23'
---

# TODO Workflow

Cross-session task backlog. Each task is self-contained — a fresh agent should
be able to pick it up from the task body alone. After completing a task, delete
its block (from the `---` above the heading to the `---` below the last line).
Instantiated from the knowledge base's `TODO_WORKFLOW_TEMPLATE.md`.

## Worklog (`worklog.jsonl`) — schema & append protocol

Each non-trivial session appends one JSON object as a new line to
`worklog.jsonl` at the repo root (append-only, oldest-first). Schema
(`schema_version: 1`): `entry_id` (`YYYY-MM-DD`, with `-s2`/`-b` suffixes for
same-day collisions), `date`, `session_id` (kb_mcp boot UUID or `null`),
`summary`, `body_markdown`. Bootstrap with `touch worklog.jsonl`. Skip the
append for trivial or purely exploratory sessions.

---

## Run the optimization & precautionary notebooks end-to-end

```yaml
status: todo
type: task
id: todo.regenerate_results
description: Execute the heavy Bayesian-optimization notebooks to regenerate their seeding CSV inputs and figures under the new layout.
owner: agent
estimate: 2h
difficulty: medium
value: high
blocked_by: []
last_checked: '2026-06-23'
```

**Context:** The reorg made all paths portable, but the long-running
optimization notebooks were not executed end-to-end. `Run_Simulations_Final.ipynb`
reloads `vax_hurts_region_seeding.csv` and `bo_results_fixed_params_seeding.csv`,
which are not yet present in `results/`.

**Preconditions:** `pip install -r requirements.txt`.

**Steps:**
1. Run `01_optimization/Run_Simulations_Final.ipynb` top to bottom; confirm it
   writes its CSVs and figures to `../results/`.
2. Run `02_precautionary/Precautionary_Principle_Analysis.ipynb`, then
   `plots.ipynb`.
3. Update `results/MANIFEST.md` with any new artifacts and paper sections.

**Verification:** `results/` contains the regenerated CSVs/figures; `plots.ipynb`
runs without `FileNotFoundError`.

**On completion:** delete this task block.

---

## Resolve unattributed figures in results/MANIFEST.md

```yaml
status: todo
type: task
id: todo.manifest_attribution
description: Attribute or remove the unattributed PNGs in results/ and de-duplicate the misspelled precautionaty_plot.png.
owner: agent
estimate: 30m
difficulty: low
value: medium
blocked_by: []
last_checked: '2026-06-23'
```

**Context:** `example_plot.png`, `output.png`, and `precautionaty_plot.png`
(misspelled) have no confirmed source notebook in `results/MANIFEST.md`.

**Preconditions:** none.

**Steps:**
1. Identify the producing notebook/cell for each, or confirm it is stale.
2. Fill the `Source notebook` / `Paper section` columns, or delete the file.
3. Delete `precautionaty_plot.png` if it duplicates `precautionary_plot.png`.

**Verification:** every file in `results/` has a MANIFEST row with a real source.

**On completion:** delete this task block.

---

## Strip Colab/Drive paths from active notebooks

```yaml
status: todo
type: task
id: todo.strip_colab_paths
description: Remove hardcoded /content/drive Colab paths so the active notebooks run unattended and pass HOUSEKEEPING Phase 2.1.
owner: agent
estimate: 30m
difficulty: low
value: high
blocked_by: []
last_checked: '2026-06-23'
```

**Context:** HOUSEKEEPING Phase 2.1 fails — `grep -rl "/content/drive" 0*/`
returns `01_optimization/Run_Simulations_Final.ipynb` and
`02_precautionary/Precautionary_Principle_Analysis.ipynb`. The Colab/Drive paths
make these notebooks fail in the Restart-and-Run-All sweep, which is why the
sweep is currently skipped.

**Preconditions:** none.

**Steps:**
1. Locate the `/content/drive` references in each notebook.
2. Replace them with the portable `../results/` (or repo-relative) paths used
   elsewhere in the strand layout.
3. Re-run `grep -rl "/content/drive" 0*/` — it should return nothing.

**Verification:** the Phase 2.1 grep is empty; both notebooks open and resolve
their I/O paths without referencing Google Drive.

**On completion:** delete this task block.

---

## Task template

````markdown
## [Task Title]

```yaml
status: todo
type: task
id: todo.[short_id]
description: One-sentence description of what this task accomplishes.
owner: agent
estimate: Xm
difficulty: [low | medium | high]
value: [low | medium | high]
blocked_by: []
last_checked: '{{YYYY-MM-DD}}'
```

**Context:** Why this task exists.

**Preconditions:** State that must be true before starting. `none` if none.

**Steps:**
1. ...

**Verification:** How to confirm completion.

**On completion:** Delete this entire task block.
````
