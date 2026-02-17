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

### Optimization
- status: active
- type: documentation
<!-- content -->
The project uses **Bayesian Optimization** (via `Run_Simulations_Final.ipynb`) to explore the parameter space and identify optimal strategies that minimize infections and deaths.

### Precautionary Principle Analysis
- status: active
- type: documentation
<!-- content -->
A specialized notebook, `Precautionary_Principle_Analysis.ipynb`, is available to study the **Precautionary Principle**. It performs two separate parameter searches to identify the **worst-case death scenarios** for "Vaccinate All" vs. "Vaccinate Vulnerable Only" strategies, allowing for a minimax analysis of the results.

## Usage
- status: active
- type: documentation
<!-- content -->

### Dependencies
- Python 3.x
- `numpy`, `matplotlib`, `pandas`, `seaborn`, `networkx`, `tqdm`, `jupyter`, `scikit-optimize`

### Running the Simulation
1.  Open `Run_Simulations_Final.ipynb` in Jupyter Notebook.
2.  The simulation is now consolidated into a single notebook that includes seeding for reproducibility.
3.  Execute the cells to run the simulation and optimization routines.
4.  Process results are visualized using `matplotlib` and saved as CSVs (e.g., `bo_results_fixed_params_seeding.csv`).

## Simulation Metrics
- status: active
- type: documentation
<!-- content -->
The simulation report now returns a 14-element vector containing key epidemiological metrics:
1.  **Step**: The final simulation step.
2.  **Max Deaths Proportion**: Peak proportion of deaths.
3.  **Max Infected**: Peak proportion of infected agents.
4.  **AUC Infected**: Area Under the Curve of infection over time.
5.  **Avg Viral Age**: Average age of the virus genome (indicating mutation).
6.  **Avg Immunity**: Average immunity level of the population.
7.  **Non-vulnerable Dead Prop**: Proportion of non-vulnerable agents who died.
8.  **Vulnerable Dead Prop**: Proportion of vulnerable agents who died.
9.  **Seed**: Random seed used for the simulation.
10. **Total Unique Infected**: Count of unique agents infected at least once.
11. **Total Infections**: Total count of infection events (including reinfections).
12. **Vul Infections**: Total infections among vulnerable agents.
13. **Non-vul Infections**: Total infections among non-vulnerable agents.
14. **Avg Reinfections**: Average number of times an agent was reinfected.
