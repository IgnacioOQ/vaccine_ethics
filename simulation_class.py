from imports import *
from agent_class import FullAgent

class Simulation:
    """
    Manages the entire simulation, including the grid, agents, and the simulation loop.
    """
    def __init__(self, grid_size=25, num_agents=600, agent_class = FullAgent, init_infected_proportion = 0.1,
                 proportion_vulnerable=0.1, vul_penalty = 0.5,
                 infection_prob=0.25, recovery_time=30, death_prob=0.05, 
                 vax_vulnerable=False,
                 vax_all=False,
                 vax_effect = 0.7,
                 viral_age_effect = 0.1,
                 immune_adaptation_effect = 0.1,
                 plot=True):
        """
        Initializes the simulation with a grid and agents.

        Parameters:
        - grid_size: The size of the grid.
        - num_agents: The number of agents to place on the grid.
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))  # Initialize an empty grid
        self.agent_class = agent_class
        self.init_infected_proportion = init_infected_proportion
        self.proportion_vulnerable = proportion_vulnerable
        self.vul_penalty = vul_penalty
        self.infection_prob=infection_prob
        self.recovery_time=recovery_time
        self.death_prob=death_prob
        self.vax_vulnerable = vax_vulnerable
        self.vax_all = vax_all
        self.vax_effect = vax_effect
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect
        self.agents = self.initialize_agents(num_agents)  # Initialize agents
        self.plot = plot
        self.s_proportions = []
        self.i_proportions = []
        self.r_proportions = []
        self.d_proportions = []

    def initialize_agents(self, num_agents):
        """
        Randomly places agents on the grid with initial states.

        Parameters:
        - num_agents: The number of agents to place on the grid.

        Returns:
        - A list of Agent objects.
        """
        agents = []
        for _ in range(num_agents):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            state = 'I' if random.random() < self.init_infected_proportion else 'S'  # Start with a few infected agents
            if random.random() < self.proportion_vulnerable:
              vul_type = 'high'
              if self.vax_vulnerable or self.vax_all:
                agent = self.agent_class(x, y, grid_size = self.grid_size, infection_prob=self.infection_prob,
                          recovery_time = self.recovery_time,death_prob=self.death_prob,vul_type=vul_type,
                          vul_penalty = self.vul_penalty, state=state, vaxxed=True, vax_effect = self.vax_effect,
                          viral_age_effect = self.viral_age_effect,
                          immune_adaptation_effect = self.immune_adaptation_effect)
              else:
                if self.vax_all:
                  agent = self.agent_class(x, y, grid_size = self.grid_size, infection_prob=self.infection_prob,
                            recovery_time = self.recovery_time,death_prob=self.death_prob,vul_type=vul_type,
                            vul_penalty = self.vul_penalty, state=state, vaxxed=True, vax_effect = self.vax_effect,
                            viral_age_effect = self.viral_age_effect,
                            immune_adaptation_effect = self.immune_adaptation_effect)
                else:
                  agent = self.agent_class(x, y, grid_size = self.grid_size, infection_prob=self.infection_prob,
                            recovery_time = self.recovery_time,death_prob=self.death_prob,vul_type=vul_type,
                            vul_penalty = self.vul_penalty, state=state, vaxxed=True, vax_effect = self.vax_effect,
                            viral_age_effect = self.viral_age_effect,
                            immune_adaptation_effect = self.immune_adaptation_effect)
            else:
              vul_type = 'low'
              if self.vax_all:
                agent = self.agent_class(x, y, grid_size = self.grid_size, infection_prob=self.infection_prob,
                          recovery_time = self.recovery_time,death_prob=self.death_prob,vul_type=vul_type,
                          vul_penalty = self.vul_penalty, state=state, vaxxed=True, vax_effect = self.vax_effect,
                          viral_age_effect = self.viral_age_effect,
                          immune_adaptation_effect = self.immune_adaptation_effect)
              else:
                agent = self.agent_class(x, y, grid_size = self.grid_size, infection_prob=self.infection_prob,
                          recovery_time = self.recovery_time,death_prob=self.death_prob,vul_type=vul_type,
                          vul_penalty = self.vul_penalty, state=state, vaxxed=True, vax_effect = self.vax_effect,
                          viral_age_effect = self.viral_age_effect,
                          immune_adaptation_effect = self.immune_adaptation_effect)

            agents.append(agent)
            # Mark the initial position on the grid
            # this is just to color the grid, the actual process uses the position in the agent class
            self.grid[x, y] = state_mapping[state]  # Mark the initial position on the grid

        return agents

    def update_agents(self):
        """
        Updates the position and state of all agents and refreshes the grid.
        """
        # Clear the grid before updating
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Update each agent and place it on the new grid position
        for agent in self.agents:
            (x, y), state = agent.update(self.agents)
            self.grid[x, y] = state_mapping[state]  # Mark the agent's new position on the grid

    def run(self, iterations,plot_grid=False):
        """
        Runs the simulation for a specified number of iterations.

        Parameters:
        - iterations: The number of simulation steps to run.
        """
        for step in range(iterations):
            self.update_agents()  # Update all agents
            if self.plot:
              clear_output(wait=False)  # True if you want it online, false if you want the end plot
              self.plot_hist()
              if plot_grid:
                self.plot_grid()
              time.sleep(0.01)  # Pause to create an animation effect

            # check states of the agents
            states = [agent.state for agent in self.agents]
            i_prop = states.count('I') / num_agents

            if i_prop == 0:
                print("Simulation stopped early: No infected agents remaining. Step: "+str(step))
                #clear_output(wait=True)
                #self.plot_hist(step+1)
                break
        # self.plot_hist()

    def plot_grid(self, step):
        """
        Plots the current state of the grid, showing the position and state of each agent.

        Parameters:
        - step: The current simulation step, used for the plot title.
        """
        plt.figure(figsize=(10, 6))  # Set the plot size

        # Prepare a grid for plotting with colors based on the agent state
        colored_grid = np.zeros((self.grid_size, self.grid_size, 3))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] != 0:
                    color = color_mapping[self.grid[i, j]]
                    colored_grid[i, j] = mcolors.to_rgb(color)

        # Display the grid with colored agents
        plt.imshow(colored_grid, interpolation='none')
        plt.grid(True, which='both', color='black', linewidth=1)
        plt.xticks(np.arange(-0.5, self.grid_size, 1), [])
        plt.yticks(np.arange(-0.5, self.grid_size, 1), [])
        plt.title(f"Simulation Step: {step}")  # Set the plot title
        plt.show()


    def plot_hist(self):

      # check states of the agents
      states = [agent.state for agent in self.agents]
      s_prop = states.count('S') / len(self.agents)
      i_prop = states.count('I') / len(self.agents)
      r_prop = states.count('R') / len(self.agents)
      d_prop = states.count('D') / len(self.agents)

      # Store the proportions for plotting later
      self.s_proportions.append(s_prop)
      self.i_proportions.append(i_prop)
      self.r_proportions.append(r_prop)
      self.d_proportions.append(d_prop)

      fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

      # Bar plot for proportions
      axs[0].bar(['S', 'I', 'R','D'], [s_prop, i_prop, r_prop,d_prop],color = ['green','red','blue','black'])
      axs[0].set_ylim(0, 1)
      axs[0].set_ylabel('Proportion')
      axs[0].set_title('Agent State Proportions')

      # Plot the populations over time
      axs[1].plot(range(len(self.s_proportions)), self.s_proportions, label='S', color='green')
      axs[1].plot(range(len(self.i_proportions)), self.i_proportions, label='I', color='red')
      axs[1].plot(range(len(self.r_proportions)), self.r_proportions, label='R', color='blue')
      axs[1].plot(range(len(self.d_proportions)), self.d_proportions, label='D', color='black')
      axs[1].set_xlim(0, max(2, len(self.s_proportions) + 1))
      axs[1].set_ylim(0, 1)
      axs[1].set_xlabel('Time Steps')
      axs[1].set_ylabel('Proportion')
      axs[1].legend()
      axs[1].set_title('State Proportions Over Time')

      plt.show()
