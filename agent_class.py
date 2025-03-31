from imports import *

class FullAgent:
    """
    Represents an agent in the simulation with a position (x, y) and a health state.
    """
    def __init__(self, x, y, grid_size = 20, infection_prob=0.3, recovery_time=10,
                 death_prob=0.01, vul_penalty = 0.5, vul_type = 'high', vaxxed = False,
                 state='S',
                 # infectivity reduction, death rate reduction, recovery rate reduction
                 # these are percent reduce, e.g. vaccinating reduces 70% (0.7) the infectivity, death, and recovery rate
                  #  viral age age and immunity do the same, times the age or immunity level respectively
                 vax_effect = 0.7,
                 viral_age_effect = 0.05,
                 immune_adaptation_effect = 0.05):
        """
        Initializes an agent with a given position and state.

        Parameters:
        - x, y: Position of the agent on the grid.
        - infection_prob: Probability of getting infected when exposed.
        - recovery_time: Time taken to recover from infection.
        - death_prob: Probability of dying if infected.
        - state: Initial health state ('S' for susceptible, 'I' for infected, 'R' for recovered, 'D' for dead).
        """
        self.x = x
        self.y = y
        self.state = state
        self.vul_type = vul_type
        if self.vul_type == 'high':
            self.infection_prob = infection_prob*(1+vul_penalty)
            self.recovery_time = recovery_time*(1+vul_penalty)
            self.death_prob = death_prob*(1+vul_penalty)
        else:
          self.infection_prob = infection_prob
          self.recovery_time = recovery_time
          self.death_prob = death_prob
        self.time_infected = 0  # Time since infection
        self.grid_size = grid_size
        # now an inner variable to make sure that we keep track how old, in terms of infections, the virus is
        self.vaxxed = vaxxed
        self.vax_effect = vax_effect
        self.viral_age = 0
        self.immunity_level = 0
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect

    def move(self):
        """
        Moves the agent randomly on the grid.

        Parameters:
        - grid_size: The size of the grid (used to ensure the agent stays within bounds).
        """
        if self.state == 'D':
            return  # Dead agents do not move

        move = random.choice(['up', 'down', 'left', 'right'])
        if move == 'up' and self.x > 0:
            self.x -= 1
        elif move == 'down' and self.x < self.grid_size - 1:
            self.x += 1
        elif move == 'left' and self.y > 0:
            self.y -= 1
        elif move == 'right' and self.y < self.grid_size - 1:
            self.y += 1

    def update(self, agents):
        """
        Updates the agent's state based on interactions with other agents.

        Parameters:
        - agents: List of all agents in the simulation.

        Returns:
        - The new position and state of the agent.
        """
        if self.state == 'D':
            return (self.x, self.y), self.state  # Dead agents do not change state

        # Infection logic: Check if the agent can be infected
        if self.state == 'S':
            random.shuffle(agents)
            for agent in agents:
                if agent.x == self.x and agent.y == self.y and agent.state == 'I':
                  viral_age = agent.viral_age
                  # as the virus infects more people it gets less infective
                  if self.vaxxed:
                    base_infection_prob = self.infection_prob*(1-self.vax_effect)
                  else:
                    base_infection_prob = self.infection_prob
                  infection_prob = max(base_infection_prob*(1 - self.viral_age_effect*agent.viral_age - self.immune_adaptation_effect*self.immunity_level),0)
                  if random.random() < infection_prob:
                      self.state = 'I'  # The agent becomes infected
                      self.viral_age = agent.viral_age + 1
                      # as the virus gets older, it gets less deadly
                      self.time_infected = 0
                      break

        # Recovery or death logic
        elif self.state == 'I':
            self.time_infected += 1
            if self.vaxxed:
              base_death_prob = self.death_prob*(1-self.vax_effect)
            else:
              base_death_prob = self.death_prob
            base_death_prob = max(base_death_prob*(1 - self.viral_age_effect*self.viral_age - self.immune_adaptation_effect*self.immunity_level),0)
            if random.random() < base_death_prob:
                self.state = 'D'  # The agent dies
            else:
              if self.vaxxed:
                base_recovery_time = self.recovery_time*(1-self.vax_effect)
              else:
                base_recovery_time = self.recovery_time
              base_recovery_time = max(base_recovery_time*(1 - self.viral_age_effect*self.viral_age - self.immune_adaptation_effect*self.immunity_level),0)
              if self.time_infected >= random.randint(int(base_recovery_time//2), int(base_recovery_time)):
                self.immunity_level += 1
                self.state = 'S'  # The agent becomes infected again

        # Move the agent to a new position
        self.move()

        return (self.x, self.y), self.state
