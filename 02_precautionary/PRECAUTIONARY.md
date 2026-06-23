# Strand 02 — Precautionary Principle

## Question

Under a precautionary (minimax) framing, which strategy is safer? This strand
runs two separate parameter searches that *maximize* the death proportion —
one for *"Vaccinate All"* and one for *"Vaccinate Vulnerable Only"* — and
compares the worst-case outcomes. The strategy with the smaller worst case is
the precautionary choice.

## Notebooks

- `Precautionary_Principle_Analysis.ipynb` — runs the two worst-case Bayesian
  searches and writes per-iteration CSVs + a summary text file to `../results/`.
- `plots.ipynb` — reconstructs `skopt` `OptimizeResult` objects from the saved
  iteration CSVs and renders convergence and comparison figures.

## Inputs / outputs

- Engine: `src/simulation_class.py`, `src/agent_class.py`, `src/imports.py`.
- `plots.ipynb` reads `../results/precautionary_vax_all_iterations_*.csv` and
  `../results/precautionary_vax_vuln_iterations_*.csv`.
- Figures: `precautionary_plot.png`, `convergence_plot.png` (see
  `results/MANIFEST.md`).
