# Vaccine Ethics Simulation
- status: active
- type: documentation
- id: vaccine_ethics.readme
- last_checked: 2026-02-15
<!-- content -->
This repository contains an **Agent-Based Model (ABM)** designed to simulate and analyze the ethical and epidemiological implications of vaccine distribution strategies. It models the spread of an infectious disease within a population of agents, incorporating factors such as vaccine hesitancy, age-based vulnerability, and social network dynamics.

## Overview
- status: active
- type: documentation
- id: vaccine_ethics.readme.overview
- last_checked: 2026-02-15
<!-- content -->
The simulation takes place on a grid where agents interact, move, and potentially transmit a disease. The core purpose is to evaluate how different parameters (e.g., vaccination rates, priority for vulnerable populations) affect overall public health outcomes like infection peaks and mortality.

## Key Features
- status: active
- type: documentation
- id: vaccine_ethics.readme.features
- last_checked: 2026-02-15
<!-- content -->
### Agents
- **Health States**: Agents transition between Susceptible (S), Infected (I), Recovered (R), and Dead (D).
- **Heterogeneity**: Agents vary by age, vulnerability to severe disease, and vaccine acceptance/hesitancy.
- **Behavior**: Agents move within the grid and interact with neighbors, facilitating disease spread.

### Simulation Environment
- **Grid-Based**: A 2D grid environment (`grid_size` is configurable).
- **Parameters**: Customizable infection probabilities, recovery times, death probabilities, and vaccination strategies.
- **Visualization**: Real-time visualization of the grid state (optional in notebooks).

## Installation & Requirements
- status: active
- type: documentation
- id: vaccine_ethics.readme.requirements
- last_checked: 2026-02-15
<!-- content -->
The project requires Python 3 and the following scientific computing libraries. You can install them via pip:

```bash
pip install numpy matplotlib pandas seaborn networkx tqdm jupyter
```

Dependencies (based on `imports.py`):
- `numpy`: Numerical operations.
- `matplotlib` & `seaborn`: Plotting and visualization.
- `pandas`: Data manipulation.
- `networkx`: Network analysis (if social graph features are used).
- `tqdm`: Progress bars.

## Usage
- status: active
- type: documentation
- id: vaccine_ethics.readme.usage
- last_checked: 2026-02-15
<!-- content -->
The primary way to run and analyze the simulations is through the provided Jupyter Notebooks.

### Running a Simulation
1.  **Launch Jupyter**:
    ```bash
    jupyter notebook
    ```
2.  **Open the Main Notebook**:
    Navigate to `Run_Simulations_Final_New.ipynb`. This notebook contains the setup to initialize the `Simulation` class, run scenarios, and visualize results.
3.  **Execute Cells**:
    Run the cells sequentially to install dependencies (if needed), import classes, and execute the simulation logic.

### Sanity Checks
You can run `sanity_check.ipynb` or `sanity_check_network.ipynb` to verify that the core mechanisms (infection spread, agent movement) are working as expected before running full-scale experiments.

## File Structure
- status: active
- type: documentation
- id: vaccine_ethics.readme.structure
- last_checked: 2026-02-15
<!-- content -->
- **`simulation_class.py`**: Defines the `Simulation` class, which manages the grid, time steps, and global logic.
- **`agent_class.py`**: Defines `FullAgent` and `FullAgent2`, encapsulating individual agent stats and behavioral rules.
- **`imports.py`**: Centralized import file for all dependencies and common constants (e.g., state mappings).
- **`Run_Simulations_Final_New.ipynb`**: The main driver notebook for conducting experiments.
- **`MD_CONVENTIONS.md`**: Specification for the Markdown-JSON schema used in this documentation.
