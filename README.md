# Vaccine Ethics Simulation

An **agent-based model (ABM)** for studying the ethical and epidemiological
trade-offs of vaccine distribution strategies — in particular, *"Vaccinate
All"* versus *"Vaccinate Vulnerable Only"* — and the conditions under which a
precautionary strategy is justified.

## Thesis

Vaccinating everyone is not always strictly better than protecting only the
vulnerable. Under an evolutionary model of viral dynamics (mutation, immune
adaptation, reinfection), broad vaccination can in some parameter regimes
*increase* total harm. This repository simulates a SEIRD process on a
networked population of heterogeneous agents to locate those regimes and to
analyze them through a precautionary-principle (minimax) lens.

The disease process is **SEIRD** (Susceptible, Exposed/Infected, Recovered,
Dead) over a grid population whose agents can migrate across nodes of an
Erdős–Rényi network. Agents differ in vulnerability (`high`/`low`), vaccination
status, viral age, and adaptive immunity.

## Repository map

This is a research repository organized as independent **strands** that share a
common simulation engine (`src/`). The unit of progress is the figure or the
claim, not the feature.

| Path | What it holds |
|:--|:--|
| `src/` | Shared simulation engine — agents, simulations, common imports. Notebooks orchestrate; modules define. |
| `01_optimization/` | Bayesian-optimization search over the parameter space for harmful-vaccination regions. |
| `02_precautionary/` | Worst-case (minimax) precautionary-principle analysis and its plots. |
| `03_validation/` | Sanity-check notebooks for the simulation dynamics (single grid + network). |
| `tests/` | Automated smoke tests for the engine (pytest). |
| `results/` | Generated figures and data CSVs, with `MANIFEST.md` mapping each artifact to its source notebook. |
| `notes/` | Paper outline, drafts, and conceptual notes. |
| `archive/` | Superseded notebooks and one-off scripts kept for provenance. |

### The engine (`src/`)

- `simulation_class.py` — `Simulation` (single isolated grid), `Simulation2`
  (network of grid nodes with agent migration), `Simulation3` (heterogeneous
  grid sizes / densities per node).
- `agent_class.py` — `FullAgent`, `FullAgent2`: per-agent health state (`S`/`I`/`R`/`D`),
  `vul_type`, `vaxxed`, `viral_age`, `immunity_level`, `infection_count`.
- `imports.py` — shared third-party imports and the state/color mappings used
  for grid visualization.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter from the repo root and open a notebook in any strand
folder. Each notebook's first cell puts `src/` on the path, so the
`from imports import *` / `from agent_class import …` imports resolve
regardless of which strand folder it lives in.

## Running

- **Optimization:** `01_optimization/Run_Simulations_Final.ipynb` runs Bayesian
  optimization over the parameter space and writes results + figures to
  `results/`.
- **Precautionary analysis:** `02_precautionary/Precautionary_Principle_Analysis.ipynb`
  runs two separate searches for the worst-case death proportion under each
  strategy; `plots.ipynb` renders the convergence/comparison figures from the
  saved iteration CSVs.
- **Validation:** the `03_validation/` notebooks exercise the dynamics directly.

All notebooks write artifacts to `../results/`. See `results/MANIFEST.md` for
the figure → notebook mapping.

## Tests

Engine smoke tests live in `tests/` and run with [pytest](https://pytest.org):

```bash
pip install pytest
pytest
```

They confirm the engine imports cleanly and that a small simulation runs and
returns a well-formed 14-element report vector. For notebooks, the honest
equivalent of a test suite is the `Restart-and-Run-All` check described in
`HOUSEKEEPING.md`.

## Simulation metrics

`generate_simulation_report` returns a 14-element vector: step, max death
proportion, max infected, AUC infected, average viral age, average immunity,
non-vulnerable dead proportion, vulnerable dead proportion, seed, total unique
infected, total infections, vulnerable infections, non-vulnerable infections,
and average reinfections.

## Repository governance

- `HOUSEKEEPING.md` — periodic repo audit checklist.
- `TODO_WORKFLOW.md` — cross-session task backlog.
- `worklog.jsonl` — append-only session history.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE`.
