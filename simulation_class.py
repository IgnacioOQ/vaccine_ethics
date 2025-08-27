from imports import *
from agent_class import FullAgent, FullAgent2

class Simulation:
    """
    Manages the entire simulation, including the grid, agents, and simulation logic.
    """

    def __init__(self, grid_size=25, num_agents=600, agent_class=FullAgent, init_infected_proportion=0.1,
                 proportion_vulnerable=0.1, vul_penalty=0.5,
                 infection_prob=0.25, recovery_time=30, death_prob=0.05,
                 vax_vulnerable=False, vax_all=False,
                 vax_effect=0.7, viral_age_effect=0.1, immune_adaptation_effect=0.1,
                 plot=True, rngseed=None, nprngseed=None):
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

        # if seed:  # seed is a boolean
        #     ss = np.random.SeedSequence()
        #     seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])
        #     self.seed = seed_int
        #     random.seed(seed_int)       # for Python's random
        #     np.random.seed(seed_int)    # for NumPy
        # else:
        #     self.seed = None

        # # Per-simulation RNGs (avoid global cross-talk)
        # self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        # self.nprng = np.random.default_rng(self.seed)  # if you later need NumPy RNG
        self.rng = rngseed
        self.nprng = nprngseed
        
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
        self.step = 0

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
            # if len(self.d_proportions) >= 50 and all(x == self.d_proportions[-1] for x in self.d_proportions[-50:]):
            #     break
            if len(self.d_proportions) >= 50:
                tail = self.d_proportions[-50:]
                if max(tail) - min(tail) < 1e-9:
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
        max_deaths_prop = max(self.d_proportions)
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
        return np.array([self.step, max_deaths_prop, max_infected, auc_infected, avg_viral_age, avg_immunity,
                         non_vulnerable_proportion_dead,vulnerable_proportion_dead,self.seed])
        
        
class Simulation2:
    def __init__(self, grid_size=20, num_agents=500, agent_class=FullAgent2,
                 init_infected_proportion=0.1, proportion_vulnerable=0.1, vul_penalty=0.5,
                 infection_prob=0.25, recovery_time=30, death_prob=0.05,
                 vax_vulnerable=False, vax_all=False, vax_effect=0.7,
                 viral_age_effect=0.1, immune_adaptation_effect=0.1,
                 plot=True, seed=True,
                 # NEW: network and migration controls
                 G=None,                # networkx.Graph; if None we’ll generate one
                 num_nodes=7,           # used only if G is None
                 epsilon_migrate=0.05   # per-step probability of hopping to a neighbor node
                 ):
        self.step = 0  # initialize step counter early
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agent_class = agent_class
        self.plot = plot

        # seeding (unchanged)
        if seed:
            ss = np.random.SeedSequence()
            seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])
            self.seed = seed_int
            random.seed(seed_int)
            np.random.seed(seed_int)
        else:
            self.seed = None
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.nprng = np.random.default_rng(self.seed)

        # disease & vax params (unchanged)
        self.init_infected_proportion = init_infected_proportion
        self.proportion_vulnerable = proportion_vulnerable
        self.vul_penalty = vul_penalty
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.death_prob = death_prob
        self.vax_vulnerable = vax_vulnerable
        self.vax_all = vax_all
        self.vax_effect = vax_effect
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect

        # NEW: network + migration knobs
        self.epsilon_migrate = epsilon_migrate
        if G is None:
            # simple connected random graph
            # regenerate until connected so agents actually can move around
            # while True:
            #     G_try = nx.erdos_renyi_graph(n=num_nodes, p=0.5, seed=self.seed)
            #     if nx.is_connected(G_try):
            #         self.G = G_try
            #         break
            # replace the while True block
            max_tries = 500
            for _ in range(max_tries):
                # vary the seed so you actually get different graphs
                s = self.rng.randrange(2**32)
                G_try = nx.erdos_renyi_graph(n=num_nodes, p=0.3, seed=s)
                if num_nodes == 1 or nx.is_connected(G_try):
                    self.G = G_try
                    break
            else:
                raise RuntimeError("Couldn't generate a connected graph. Increase p or num_nodes.")
        else:
            self.G = G

        # NEW: per-node grids (each node has its own grid)
        self.node_grids = {n: np.zeros((grid_size, grid_size)) for n in self.G.nodes}

        # Initialize agents with node assignments
        self.agents = self.initialize_agents(num_agents)

        # tracking (unchanged)
        self.s_proportions, self.i_proportions, self.r_proportions, self.d_proportions = [], [], [], []

    def initialize_agents(self, num_agents):
        agents = []
        nodes = list(self.G.nodes)
        for _ in range(num_agents):
            node_id = self.rng.choice(nodes)
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
                rng=self.rng,
                node_id=node_id,        # NEW
            )
            agents.append(agent)
            self.node_grids[node_id][x, y] = state_mapping[state]
        return agents

    def _rebuild_node_grids(self):
        # Clear and rebuild all node grids from agent positions
        for n in self.node_grids:
            self.node_grids[n][:] = 0
        for a in self.agents:
            self.node_grids[a.node_id][a.x, a.y] = state_mapping[a.state]

    def update_agents(self):
        """
        1) Update within-node disease & local movement.
        2) Perform cross-node migration with probability epsilon.
        3) Rebuild per-node grids.
        """
        # Group agents by current node
        node_to_agents = defaultdict(list)
        for a in self.agents:
            node_to_agents[a.node_id].append(a)

        # 1) Within-node updates (infection/movement as before, but scoped to same node)
        for node_id, node_agents in node_to_agents.items():
            for agent in node_agents:
                (x, y), state = agent.update(node_agents)  # pass only same-node agents

        # 2) Cross-node migration at end of step
        for agent in self.agents:
            if agent.state == 'D':
                continue  # dead agents don't migrate
            if self.rng.random() < self.epsilon_migrate:
                nbrs = list(self.G.neighbors(agent.node_id))
                if nbrs:
                    new_node = self.rng.choice(nbrs)
                    agent.migrate_to(new_node, self.grid_size)

        # 3) Refresh per-node grids
        self._rebuild_node_grids()

    def run(self, iterations=1000, plot_grid=False):
        for step in range(iterations):
            self.step = step
            print(step, end='\r')  # Print step number in place
            self.update_agents()

            states = [a.state for a in self.agents]
            s_prop = states.count('S') / len(self.agents)
            i_prop = states.count('I') / len(self.agents)
            r_prop = states.count('R') / len(self.agents)
            d_prop = states.count('D') / len(self.agents)

            self.s_proportions.append(s_prop)
            self.i_proportions.append(i_prop)
            self.r_proportions.append(r_prop)
            self.d_proportions.append(d_prop)

            if i_prop == 0:
                break
            if len(self.d_proportions) >= 50 and all(x == self.d_proportions[-1] for x in self.d_proportions[-50:]):
                break

        if self.plot:
            clear_output(wait=False)
            self.plot_hist()
            # (Optional) You can add a small per-node grid panel plot if you want
            time.sleep(0.01)
            
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
        max_deaths_prop = max(self.d_proportions)
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
        return np.array([self.step, max_deaths_prop, max_infected, auc_infected, avg_viral_age, avg_immunity,
                         non_vulnerable_proportion_dead,vulnerable_proportion_dead,self.seed])
        
        
class Simulation3:
    def __init__(self, grid_size=20, num_agents=500, agent_class=FullAgent2,
                 init_infected_proportion=0.1, proportion_vulnerable=0.1, vul_penalty=0.5,
                 infection_prob=0.25, recovery_time=30, death_prob=0.05,
                 vax_vulnerable=False, vax_all=False, vax_effect=0.7,
                 viral_age_effect=0.1, immune_adaptation_effect=0.1,
                 plot=True, seed=True,
                 # network + migration
                 G=None, num_nodes=7, epsilon_migrate=0.05,
                 # NEW knobs for per-node variation
                 grid_jitter_frac=0.20,      # each node grid ~ Uniform[(1-δ),(1+δ)] * grid_size
                 agents_jitter_frac=0.20,    # each node agents ~ Uniform[(1-δ),(1+δ)] * num_agents
                 min_grid_size=5):
        """
        Each node n gets:
          - its own grid size: node_grid_sizes[n] ~ around grid_size
          - its own initial agent count: node_agent_counts[n] ~ around num_agents
        Total agents ~= num_nodes * num_agents (± jitter).
        """
        self.step = 0
        self.base_grid_size = grid_size
        self.per_node_mean_agents = num_agents
        self.agent_class = agent_class
        self.plot = plot

        # seeding (kept as your bool design)
        if seed:
            ss = np.random.SeedSequence()
            seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])
            self.seed = seed_int
            random.seed(seed_int)
            np.random.seed(seed_int)
        else:
            self.seed = None
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.nprng = np.random.default_rng(self.seed)

        # disease/vax params
        self.init_infected_proportion = init_infected_proportion
        self.proportion_vulnerable = proportion_vulnerable
        self.vul_penalty = vul_penalty
        self.infection_prob = infection_prob
        self.recovery_time = recovery_time
        self.death_prob = death_prob
        self.vax_vulnerable = vax_vulnerable
        self.vax_all = vax_all
        self.vax_effect = vax_effect
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect

        # graph
        self.epsilon_migrate = epsilon_migrate
        if G is None:
            max_tries = 500
            for _ in range(max_tries):
                s = self.rng.randrange(2**32)
                G_try = nx.erdos_renyi_graph(n=num_nodes, p=0.3, seed=s)
                if num_nodes == 1 or nx.is_connected(G_try):
                    self.G = G_try
                    break
            else:
                raise RuntimeError("Couldn't generate a connected graph. Increase p or num_nodes.")
        else:
            self.G = G

        self.nodes = list(self.G.nodes)

        # --- NEW: per-node grid sizes and agent counts ---
        self.node_grid_sizes = {}
        self.node_agent_counts = {}
        for n in self.nodes:
            # grid size jitter
            low_g  = max(min_grid_size, int(round(self.base_grid_size * (1 - grid_jitter_frac))))
            high_g = max(min_grid_size, int(round(self.base_grid_size * (1 + grid_jitter_frac))))
            gsize = self.rng.randint(low_g, high_g)
            self.node_grid_sizes[n] = gsize

            # per-node initial agents jitter
            low_a  = max(1, int(round(self.per_node_mean_agents * (1 - agents_jitter_frac))))
            high_a = max(1, int(round(self.per_node_mean_agents * (1 + agents_jitter_frac))))
            acount = self.rng.randint(low_a, high_a)
            self.node_agent_counts[n] = acount

        # per-node grids (dtype uint8 for compactness)
        self.node_grids = {n: np.zeros((self.node_grid_sizes[n], self.node_grid_sizes[n]),
                                       dtype=np.uint8) for n in self.nodes}

        # Initialize agents per node (sum over nodes)
        self.agents = self._initialize_agents_per_node()

        # tracking
        self.s_proportions, self.i_proportions, self.r_proportions, self.d_proportions = [], [], [], []

    # ----------- internals -----------
    def _spawn_agent(self, node_id):
        gsize = self.node_grid_sizes[node_id]
        x = self.rng.randint(0, gsize - 1)
        y = self.rng.randint(0, gsize - 1)
        state = 'I' if self.rng.random() < self.init_infected_proportion else 'S'
        vul_type = 'high' if self.rng.random() < self.proportion_vulnerable else 'low'
        vaxxed = self.vax_all or (vul_type == 'high' and self.vax_vulnerable)

        # IMPORTANT: set per-agent grid_size to this node's size
        agent = self.agent_class(
            x, y,
            grid_size=gsize,
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
            rng=self.rng,
            node_id=node_id,  # expected by your FullAgent2
        )
        return agent

    def _initialize_agents_per_node(self):
        agents = []
        for n in self.nodes:
            for _ in range(self.node_agent_counts[n]):
                a = self._spawn_agent(n)
                agents.append(a)
                self.node_grids[n][a.x, a.y] = state_mapping[a.state]
        return agents

    def _rebuild_node_grids(self):
        for n in self.node_grids:
            self.node_grids[n][:] = 0
        for a in self.agents:
            self.node_grids[a.node_id][a.x, a.y] = state_mapping[a.state]

    def _migrate_agent(self, agent, new_node):
        """Move agent to a neighbor node with different grid size."""
        agent.node_id = new_node
        # switch to the new node's grid size and randomize position inside it
        new_gsize = self.node_grid_sizes[new_node]
        agent.grid_size = new_gsize
        agent.x = self.rng.randint(0, new_gsize - 1)
        agent.y = self.rng.randint(0, new_gsize - 1)

    # ----------- main loop -----------
    def update_agents(self):
        # group by node
        node_to_agents = defaultdict(list)
        for a in self.agents:
            node_to_agents[a.node_id].append(a)

        # within-node updates
        for node_id, node_agents in node_to_agents.items():
            # optional: shuffle to reduce order bias
            self.rng.shuffle(node_agents)
            for agent in node_agents:
                agent.update(node_agents)

        # cross-node migration
        for agent in self.agents:
            if agent.state == 'D':
                continue
            if self.rng.random() < self.epsilon_migrate:
                nbrs = list(self.G.neighbors(agent.node_id))
                if nbrs:
                    new_node = self.rng.choice(nbrs)
                    self._migrate_agent(agent, new_node)

        # refresh grids
        self._rebuild_node_grids()

    def run(self, iterations=1000, plot_grid=False):
        for step in range(iterations):
            self.step = step
            print(step, end='\r', flush=True)
            self.update_agents()

            states = [a.state for a in self.agents]
            s_prop = states.count('S') / len(self.agents)
            i_prop = states.count('I') / len(self.agents)
            r_prop = states.count('R') / len(self.agents)
            d_prop = states.count('D') / len(self.agents)

            self.s_proportions.append(s_prop)
            self.i_proportions.append(i_prop)
            self.r_proportions.append(r_prop)
            self.d_proportions.append(d_prop)

            if i_prop == 0:
                break
            if len(self.d_proportions) >= 50 and all(x == self.d_proportions[-1] for x in self.d_proportions[-50:]):
                break

        if self.plot:
            clear_output(wait=False)
            if hasattr(self, "plot_hist"):
                self.plot_hist()
            time.sleep(0.01)

    # --- keep your plot_hist / report methods as-is, they work unchanged ---
    def plot_hist(self):
        def smooth(data, window_size=5):
            pad = window_size // 2
            padded = np.pad(data, (pad, pad), mode='edge')
            return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        s_prop = self.s_proportions[-1]
        i_prop = self.i_proportions[-1]
        d_prop = self.d_proportions[-1]
        axs[0].bar(['S', 'I', 'D'], [s_prop, i_prop, d_prop], color=['green', 'red', 'black'])
        axs[0].set_ylim(0, 1); axs[0].set_ylabel('Proportion'); axs[0].set_title('Final Agent State Proportions')

        window_size = 5
        s_series = smooth(self.s_proportions, window_size)
        i_series = smooth(self.i_proportions, window_size)
        d_series = smooth(self.d_proportions, window_size)
        axs[1].plot(range(len(s_series)), s_series, label='S', color='green')
        axs[1].plot(range(len(i_series)), i_series, label='I', color='red')
        axs[1].plot(range(len(d_series)), d_series, label='D', color='black')
        axs[1].set_xlim(0, max(2, len(s_series) + 1)); axs[1].set_ylim(0, 1)
        axs[1].set_xlabel('Time Steps'); axs[1].set_ylabel('Proportion'); axs[1].legend()
        axs[1].set_title('Smoothed State Proportions Over Time')
        plt.show()

    def generate_simulation_report(self):
        max_deaths_prop = max(self.d_proportions)
        max_infected = max(self.i_proportions)
        time_steps = range(len(self.i_proportions))
        auc_infected = np.trapz(self.i_proportions, x=time_steps)
        avg_viral_age = np.mean([agent.viral_age for agent in self.agents])
        avg_immunity = np.mean([agent.immunity_level for agent in self.agents])

        total_non_vulnerable = len([agent for agent in self.agents if agent.vul_type == 'low'])
        who_died = [agent for agent in self.agents if agent.state == 'D']
        non_vulnerable_dead = len([agent for agent in who_died if agent.vul_type == 'low'])
        non_vulnerable_proportion_dead = non_vulnerable_dead / total_non_vulnerable if total_non_vulnerable > 0 else 0

        total_vulnerable = len([agent for agent in self.agents if agent.vul_type == 'high'])
        vulnerable_dead = len([agent for agent in who_died if agent.vul_type == 'high'])
        vulnerable_proportion_dead = vulnerable_dead / total_vulnerable if total_vulnerable > 0 else 0

        return np.array([self.step, max_deaths_prop, max_infected, auc_infected,
                         avg_viral_age, avg_immunity,
                         non_vulnerable_proportion_dead, vulnerable_proportion_dead,
                         self.seed])
