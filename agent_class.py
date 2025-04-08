from imports import *

class FullAgent:
    """
    Represents an agent in the simulation with a position (x, y) and a health state.
    The agent can move around a grid, interact with other agents, get infected,
    recover, or die depending on certain probabilities and health factors.
    """

    def __init__(self, x, y, grid_size=20, infection_prob=0.3, recovery_time=10,
                 death_prob=0.01, vul_penalty=0.5, vul_type='high', vaxxed=False,
                 state='S',
                 # infectivity reduction, death rate reduction, recovery rate reduction
                 # these are percent reductions, e.g. vaccination reduces 70% (0.7)
                 # of the infectivity, death, and recovery rate.
                 # Viral age and immunity apply the same reductions, scaled by level
                 vax_effect=0.7,
                 viral_age_effect=0.05,
                 immune_adaptation_effect=0.05):
        """
        Initializes an agent with a given position and state.

        Parameters:
        - x, y: Initial position of the agent on the grid.
        - grid_size: Size of the simulation grid (default: 20).
        - infection_prob: Base probability of becoming infected upon exposure.
        - recovery_time: Time steps required to recover from infection.
        - death_prob: Probability of dying from an infection.
        - vul_penalty: Multiplier applied to risk factors for vulnerable agents.
        - vul_type: 'high' or other, determines whether the agent is vulnerable.
        - vaxxed: Boolean, whether the agent is vaccinated.
        - state: Initial health state ('S'=susceptible, 'I'=infected, 'R'=recovered, 'D'=dead).
        - vax_effect: Reduction applied due to vaccination (0.7 = 70% reduction).
        - viral_age_effect: Reduction per unit of viral age.
        - immune_adaptation_effect: Reduction per unit of immunity level.
        """
        self.x = x
        self.y = y
        self.state = state
        self.vul_type = vul_type

        # Apply vulnerability penalties if the agent is 'high risk'
        if self.vul_type == 'high':
            self.infection_prob = min(infection_prob * (1 + vul_penalty),1)
            self.recovery_time = min(recovery_time * (1 + vul_penalty),1)
            self.death_prob = min(death_prob * (1 + vul_penalty),1)
        else:
            self.infection_prob = infection_prob
            self.recovery_time = recovery_time
            self.death_prob = death_prob

        self.time_infected = 0  # Tracks how long the agent has been infected
        self.grid_size = grid_size

        # Virus tracking variables
        self.vaxxed = vaxxed
        self.vax_effect = vax_effect
        self.viral_age = 0  # Increments each time the virus passes through an agent
        self.immunity_level = 0  # Tracks the agent's learned immunity from past infections
        self.viral_age_effect = viral_age_effect
        self.immune_adaptation_effect = immune_adaptation_effect

    def move(self):
        """
        Randomly moves the agent one step in the grid (up, down, left, or right).
        Dead agents do not move.
        """
        if self.state == 'D':
            return  # Dead agents cannot move

        move = random.choice(['up', 'down', 'left', 'right'])

        # Ensure agent stays within grid boundaries
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
        Updates the agent's state based on contact with other agents and internal state.

        Parameters:
        - agents: List of all agents in the simulation.

        Returns:
        - Tuple of the agent's new (x, y) position and health state.
        """
        if self.state == 'D':
            return (self.x, self.y), self.state  # Dead agents do not update

        # Check for possible infection from nearby infected agents
        if self.state == 'S':
            random.shuffle(agents)  # Randomize interaction order
            for agent in agents:
                if agent.x == self.x and agent.y == self.y and agent.state == 'I':
                    viral_age = agent.viral_age  # Get how many times the virus has passed hosts

                    # Apply vaccine effect to infection probability if vaccinated
                    if self.vaxxed:
                        base_infection_prob = self.infection_prob * (1 - self.vax_effect)
                    else:
                        base_infection_prob = self.infection_prob

                    # Further reduce infection probability based on viral age and immunity
                    infection_prob = max(
                        base_infection_prob *
                        (1 - self.immune_adaptation_effect * self.immunity_level 
                         + self.viral_age_effect * agent.viral_agent), # - self.viral_age_effect * agent.viral_age
                        0
                    )

                    # Infect if random chance succeeds
                    if random.random() < infection_prob:
                        self.state = 'I'
                        self.viral_age = agent.viral_age + 1  # Inherit and increment virus age
                        self.time_infected = 0
                        break  # Only one infection attempt per time step

        elif self.state == 'I':
            # Progress infection time
            self.time_infected += 1

            # Calculate adjusted death probability
            if self.vaxxed:
                base_death_prob = self.death_prob * (1 - self.vax_effect)
            else:
                base_death_prob = self.death_prob

            base_death_prob = max(
                base_death_prob *
                (1 - self.viral_age_effect * self.viral_age - self.immune_adaptation_effect * self.immunity_level),
                0
            )

            # Determine if the agent dies
            if random.random() < base_death_prob:
                self.state = 'D'
            else:
                # If not dead, determine if the agent recovers
                if self.vaxxed:
                    base_recovery_time = self.recovery_time * (1 - self.vax_effect)
                else:
                    base_recovery_time = self.recovery_time

                base_recovery_time = max(
                    base_recovery_time *
                    (1 - self.immune_adaptation_effect * self.immunity_level), # - self.viral_age_effect * self.viral_age 
                    0
                )

                # Recover if infection time exceeds randomized threshold
                if self.time_infected >= random.randint(int(base_recovery_time // 2), int(base_recovery_time)):
                    self.immunity_level += 1  # Gain some immunity
                    self.state = 'S'  # Become susceptible again (not truly recovered)

        # Move the agent regardless of health state (except dead)
        self.move()

        return (self.x, self.y), self.state