# Results Manifest

Maps each generated artifact to the notebook that produces it, for figure
traceability. Update this table whenever a notebook starts producing a new
figure or CSV. `*.png`, `*.html`, and `*.csv` here are tracked; regenerable
`*.pkl` / `*.npy` are gitignored.

## Figures

| File | Source notebook | Paper section |
|:--|:--|:--|
| `vax_hurts_3d_plot.png` | `01_optimization/Run_Simulations_Final.ipynb` | TBD |
| `interactive_vax_hurts_plot.html` | `01_optimization/Run_Simulations_Final.ipynb` | TBD |
| `precautionary_plot.png` | `02_precautionary/plots.ipynb` | TBD |
| `convergence_plot.png` | `02_precautionary/plots.ipynb` | TBD |
| `example_plot.png` | (unattributed — confirm source) | TBD |
| `output.png` | (unattributed — confirm source) | TBD |
| `precautionaty_plot.png` | (misspelled duplicate of `precautionary_plot.png`? — confirm/remove) | TBD |

## Data

| File | Source notebook | Notes |
|:--|:--|:--|
| `precautionary_vax_all_iterations_20260303-132721.csv` | `02_precautionary/Precautionary_Principle_Analysis.ipynb` | Read by `plots.ipynb`. |
| `precautionary_vax_vuln_iterations_20260303-132721.csv` | `02_precautionary/Precautionary_Principle_Analysis.ipynb` | Read by `plots.ipynb`. |
| `bo_results_fixed_params.csv` | `01_optimization/Run_Simulations_Final.ipynb` | Optimization sweep output. |
| `bo_results_fixed_params_seeding.csv` | `01_optimization/Run_Simulations_Final.ipynb` | Seeding input reloaded by the optimization notebook. |
| `vax_hurts_region_seeding.csv` | `01_optimization/Run_Simulations_Final.ipynb` | Seeding input reloaded by the optimization notebook. |

## Open items

- Several PNGs above are unattributed; confirm which notebook/cell produced
  them, or remove if stale.
- `precautionaty_plot.png` appears to be a misspelled duplicate of
  `precautionary_plot.png` — confirm and delete one.
