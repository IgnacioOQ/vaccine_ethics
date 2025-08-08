from imports import *
from agent_class import FullAgent

class Simulation:
    """
    Manages the entire simulation, including the grid, agents, and simulation logic.
    """

    def __init__(self, grid_size=25, num_agents=600, agent_class=FullAgent, init_infected_proportion=0.1,
                 proportion_vulnerable=0.1, vul_penalty=0.5,
                 infection_prob=0.25, recovery_time=30, death_prob=0.05,
                 vax_vulnerable=False, vax_all=False,
                 vax_effect=0.7, viral_age_effect=0.1, immune_adaptation_effect=0.1,
                 plot=True, seed=True):
        """
        Initializes the simulation grid and the agents.

        Parameters:
        - grid_size: Size of the grid (NxN).
        - num_agents: Number of agents in the simulation.
        - agent_class: Class used to instantiate agents (default: FullAgent).
        - init_infected_proportion: Proportion of agents initially infected.
        - proportion_vulnerable: Proportion of vulnerable agents (higher risk).
        - vul_penalty: Penalty multiplier for vulnerable agents.
        - infection_prob: Base probability of infection.
        - recovery_time: Base recovery time for infected agents.
        - death_prob: Base probability of death due to infection.
        - vax_vulnerable: Whether to vaccinate vulnerable agents.
        - vax_all: Whether to vaccinate all agents.
        - vax_effect: Effectiveness of vaccination (reduction factor).
        - viral_age_effect: Infectivity/death/recovery reduction per viral age.
        - immune_adaptation_effect: Same reduction per immunity level.
        - plot: Whether to plot results at the end.
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # Empty grid
        self.agent_class = agent_class

        if seed:  # seed is a boolean
            ss = np.random.SeedSequence()
            seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])
            self.seed = seed_int
            random.seed(seed_int)       # for Python's random
            np.random.seed(seed_int)    # for NumPy
        else:
            self.seed = None

        # Per-simulation RNGs (avoid global cross-talk)
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.nprng = np.random.default_rng(self.seed)  # if you later need NumPy RNG

        # Disease and agent behavior parameters
        self.init_infected_proportion = init_infected_proportion
        self.proportion_vulnerable = proportion_vulnerable
        self.vul_penalty = vul_penalty
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.death_prob = death_prob

        # Vaccination settings
        self.vax_vulnerable = vax_vulnerable
        self.vax_all = vax_all
        self.vax_effect = vax_effect

        # Evolutionary factors
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect

        # Initialize agents
        self.agents = self.initialize_agents(num_agents)

        # State tracking and plotting
        self.plot = plot
        self.s_proportions = []
        self.i_proportions = []
        self.r_proportions = []
        self.d_proportions = []

    def initialize_agents(self, num_agents):
        agents = []
        for _ in range(num_agents):
            x = self.rng.randint(0, self.grid_size - 1)
            y = self.rng.randint(0, self.grid_size - 1)
            state = 'I' if self.rng.random() < self.init_infected_proportion else 'S'
            vul_type = 'high' if self.rng.random() < self.proportion_vulnerable else 'low'
            vaxxed = self.vax_all or (vul_type == 'high' and self.vax_vulnerable)

            agent = self.agent_class(
                x, y,
                grid_size=self.grid_size,
                infection_prob=self.infection_prob,
                recovery_time=self.recovery_time,
                death_prob=self.death_prob,
                vul_type=vul_type,
                vul_penalty=self.vul_penalty,
                state=state,
                vaxxed=vaxxed,
                vax_effect=self.vax_effect,
                viral_age_effect=self.viral_age_effect,
                immune_adaptation_effect=self.immune_adaptation_effect,
                rng=self.rng  # <<< pass per-sim RNG
            )

            agents.append(agent)
            self.grid[x, y] = state_mapping[state]
        return agents
    
    def update_agents(self):
        """
        Updates agent positions and states, and refreshes the grid.
        """
        self.grid = np.zeros((self.grid_size, self.grid_size))  # Clear grid

        for agent in self.agents:
            (x, y), state = agent.update(self.agents)
            self.grid[x, y] = state_mapping[state]

    def run(self, iterations=1000, plot_grid=False):
        """
        Runs the simulation for a given number of iterations.

        Parameters:
        - iterations: Number of time steps to simulate.
        - plot_grid: Whether to show the final grid layout.
        """
        for step in range(iterations):
            self.step = step
            self.update_agents()

            # Count states
            states = [agent.state for agent in self.agents]
            s_prop = states.count('S') / len(self.agents)
            i_prop = states.count('I') / len(self.agents)
            r_prop = states.count('R') / len(self.agents)
            d_prop = states.count('D') / len(self.agents)

            # Store proportions
            self.s_proportions.append(s_prop)
            self.i_proportions.append(i_prop)
            self.r_proportions.append(r_prop)
            self.d_proportions.append(d_prop)

            # Stop if no infected remain
            if i_prop == 0:
                break
            
            # Stop if there is no progress in proportions of deaths
            if len(self.d_proportions) >= 50 and all(x == self.d_proportions[-1] for x in self.d_proportions[-50:]):
                break
            
        if self.plot:
            clear_output(wait=False)
            self.plot_hist()
            if plot_grid:
                self.plot_grid(step)
            time.sleep(0.01)

    def plot_grid(self, step):
        """
        Visualizes the agent states on the grid at a specific simulation step.

        Parameters:
        - step: Simulation step number.
        """
        plt.figure(figsize=(10, 6))
        colored_grid = np.zeros((self.grid_size, self.grid_size, 3))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] != 0:
                    color = color_mapping[self.grid[i, j]]
                    colored_grid[i, j] = mcolors.to_rgb(color)

        plt.imshow(colored_grid, interpolation='none')
        plt.grid(True, which='both', color='black', linewidth=1)
        plt.xticks(np.arange(-0.5, self.grid_size, 1), [])
        plt.yticks(np.arange(-0.5, self.grid_size, 1), [])
        plt.title(f"Simulation Step: {step}")
        plt.show()

    # def plot_hist(self):
    #     """
    #     Plots the final state proportions and how states evolved over time.
    #     """
    #     fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    #     # Final proportions (bar chart)
    #     s_prop = self.s_proportions[-1]
    #     i_prop = self.i_proportions[-1]
    #     r_prop = self.r_proportions[-1]
    #     d_prop = self.d_proportions[-1]

    #     axs[0].bar(['S', 'I', 'D'], [s_prop, i_prop, d_prop], color=['green', 'red', 'black'])
    #     axs[0].set_ylim(0, 1)
    #     axs[0].set_ylabel('Proportion')
    #     axs[0].set_title('Final Agent State Proportions')

    #     # Time evolution (line chart)
    #     axs[1].plot(range(len(self.s_proportions)), self.s_proportions, label='S', color='green')
    #     axs[1].plot(range(len(self.i_proportions)), self.i_proportions, label='I', color='red')
    #     # axs[1].plot(range(len(self.r_proportions)), self.r_proportions, label='R', color='blue')
    #     axs[1].plot(range(len(self.d_proportions)), self.d_proportions, label='D', color='black')
    #     axs[1].set_xlim(0, max(2, len(self.s_proportions) + 1))
    #     axs[1].set_ylim(0, 1)
    #     axs[1].set_xlabel('Time Steps')
    #     axs[1].set_ylabel('Proportion')
    #     axs[1].legend()
    #     axs[1].set_title('State Proportions Over Time')

    #     plt.show()

    def plot_hist(self):
        """
        Plots the final state proportions and how states evolved over time (smoothed).
        """
        def smooth(data, window_size=5):
            pad = window_size // 2
            padded = np.pad(data, (pad, pad), mode='edge')
            return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Final proportions (bar chart)
        s_prop = self.s_proportions[-1]
        i_prop = self.i_proportions[-1]
        d_prop = self.d_proportions[-1]

        axs[0].bar(['S', 'I', 'D'], [s_prop, i_prop, d_prop], color=['green', 'red', 'black'])
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Proportion')
        axs[0].set_title('Final Agent State Proportions')

        # Smooth time series with padding
        window_size = 5
        s_series = smooth(self.s_proportions, window_size)
        i_series = smooth(self.i_proportions, window_size)
        d_series = smooth(self.d_proportions, window_size)

        # Time evolution (line chart)
        axs[1].plot(range(len(s_series)), s_series, label='S', color='green')
        axs[1].plot(range(len(i_series)), i_series, label='I', color='red')
        axs[1].plot(range(len(d_series)), d_series, label='D', color='black')
        axs[1].set_xlim(0, max(2, len(s_series) + 1))
        axs[1].set_ylim(0, 1)
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Proportion')
        axs[1].legend()
        axs[1].set_title('Smoothed State Proportions Over Time')

        plt.show()

    def generate_simulation_report(self):
        """
        Generates a summary report of the simulation results.

        Returns:
        - NumPy array containing:
          [last_step, max_deaths, peak_infection, infection_auc, avg_viral_age, avg_immunity]
        """
        dead_count = max(self.d_proportions)
        max_infected = max(self.i_proportions)
        time_steps = range(len(self.i_proportions))
        auc_infected = np.trapz(self.i_proportions, x=time_steps)  # Area under curve

        avg_viral_age = np.mean([agent.viral_age for agent in self.agents])
        avg_immunity = np.mean([agent.immunity_level for agent in self.agents])

        total_non_vulnerable = len([agent for agent in self.agents if agent.vul_type == 'low'])
        who_died = [agent for agent in self.agents if agent.state == 'D']
        non_vulnerable_dead = len([agent for agent in who_died if agent.vul_type == 'low'])
        non_vulnerable_proportion_dead = non_vulnerable_dead / total_non_vulnerable if total_non_vulnerable > 0 else 0
        # we later want to compare between the vax all vs vax vulnerable
        total_vulnerable = len([agent for agent in self.agents if agent.vul_type == 'high'])
        vulnerable_dead = len([agent for agent in who_died if agent.vul_type == 'high'])
        vulnerable_proportion_dead = vulnerable_dead / total_vulnerable if total_vulnerable > 0 else 0
        return np.array([self.step, dead_count, max_infected, auc_infected, avg_viral_age, avg_immunity,
                         non_vulnerable_proportion_dead,vulnerable_proportion_dead,self.seed])