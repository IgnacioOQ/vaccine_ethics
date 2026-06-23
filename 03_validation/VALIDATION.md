# Strand 03 — Validation

## Purpose

Sanity checks on the simulation engine — confirming the SEIRD dynamics, agent
attributes, and network migration behave as intended before the results are
trusted by the optimization and precautionary strands.

## Notebooks

- `sanity_check.ipynb` — single-grid dynamics and report-vector checks.
- `sanity_check_2.ipynb` — multi-run aggregation and distribution checks.
- `sanity_check_network.ipynb` — networked (`Simulation2`/`Simulation3`)
  multi-node dynamics with agent migration.

## Inputs / outputs

- Engine: `src/simulation_class.py`, `src/agent_class.py`, `src/imports.py`.
- These notebooks are diagnostic; they are not the source of any paper figure.
