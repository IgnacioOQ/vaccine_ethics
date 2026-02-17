# Vaccine Ethics Simulation
- status: active
- type: documentation
- id: vaccine_ethics.readme
- last_checked: 2026-02-17
<!-- content -->
This repository contains an **Agent-Based Model (ABM)** designed to simulate and analyze the ethical and epidemiological implications of vaccine distribution strategies.

## Overview
- status: active
- type: documentation
<!-- content -->
The simulation models the spread of an infectious disease (SEIRD model: Susceptible, Exposed/Infected, Recovered, Dead) within a population of agents. It incorporates factors such as:
- **Vaccine Hesitancy & Availability**: Strategies for vaccinating the entire population vs. prioritizing vulnerable groups.
- **Age-Based Vulnerability**: Agents have different risk profiles (`high` vs. `low` vulnerability).
- **Social Network Dynamics**: Agents interact on a grid and can migrate between nodes in a network (represented as an Erdos-Renyi graph).
- **Evolutionary Viral Dynamics**: Viral age and immune adaptation affect transmission and recovery rates.

## Key Components
- status: active
- type: documentation
<!-- content -->

### Simulation
The core logic is defined in `simulation_class.py`.
- **Simulation**: A single isolated grid environment.
- **Simulation2**: A network of multiple grid-based nodes where agents can migrate.
- **Simulation3**: Similar to `Simulation2` but supports heterogeneous grid sizes and population densities per node.

### Agents
Defined in `agent_class.py`, agents (`FullAgent`, `FullAgent2`) possess individual states and attributes:
- **Health State**: `S` (Susceptible), `I` (Infected), `R` (Recovered), `D` (Dead).
- **Attributes**:
    - `vul_type`: Vulnerability level (`high` or `low`).
    - `vaxxed`: Vaccination status.
    - `viral_age`: Tracks viral mutations/passage.
    - `immunity_level`: Adaptive immunity gained from recovery.

## Optimization
- status: active
- type: documentation
<!-- content -->
The project uses **Bayesian Optimization** (via `Run_Simulations_Final_New.ipynb`) to explore the parameter space and identify optimal strategies that minimize infections and deaths.

## Usage
- status: active
- type: documentation
<!-- content -->

### Dependencies
- Python 3.x
- `numpy`, `matplotlib`, `pandas`, `seaborn`, `networkx`, `tqdm`, `jupyter`

### Running the Simulation
1.  Open `Run_Simulations_Final_New.ipynb` in Jupyter Notebook.
2.  Execute the cells to run the simulation and optimization routines.
3.  Process results are visualized using `matplotlib` and saved as CSVs (e.g., `bo_results_fixed_params.csv`).
