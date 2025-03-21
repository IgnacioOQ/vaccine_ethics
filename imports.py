import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from IPython.display import clear_output
import tqdm as tqdm

# State mapping for grid visualization with colors
state_mapping = {
    'S': 1,  # Susceptible - green
    'I': 2,  # Infected - red
    'R': 3,  # Recovered - blue
    'D': 4   # Dead - black
}

# Mapping the numerical state to colors for plotting
color_mapping = {
    1: 'green',  # Susceptible agents will be shown in green
    2: 'red',    # Infected agents will be shown in red
    3: 'blue',   # Recovered agents will be shown in blue
    4: 'yellow'   # Dead agents will be shown in black
}
