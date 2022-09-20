import json
import random

import gym
import gym.spaces

from blockland import BlocklandEnv


class SingleAgentControlledBlocklandEnv(BlocklandEnv):

    def __init__(self, level, agent_id, policies={}, random_walk=False):
        """Blockland environment with fixed policy for all agents except one.

        Args:
            level: Level configuration.
            agent_id: ID of the controlled agent.
            policies: Dictionary of policies mapping agent ID to their policy.
            random_walk: Random agent behavior. True for random walk, False for natural walk.
        """
        super().__init__(level)
        self.agent_id = agent_id
        self.policies = policies
        self.random_walk = random_walk
        agent_index = -1
        for i, agent in enumerate(self.agents):
            if agent.id == self.agent_id:
                self.agent_index = i
        self.full_action_space = self.action_space
        self.action_space = gym.spaces.Discrete(self.full_action_space.nvec[agent_index])
        if not self.random_walk:
            self.init_natural_movement()

    def step(self, action):
        """Take a step through the environment.

        Args:
            action: Action taken by the controlled agent.
        """
        actions = []
        for i, agent in enumerate(self.agents):
            if agent.id == self.agent_id:
                actions.append(action)
            elif agent.id in self.policies and self.policies[agent.id] is not None:
                a, _ = self.policies[agent.id].predict(list(self.get_states().values()))
                actions.append(a)
            elif self.random_walk:
                actions.append(random.randint(0, self.full_action_space.nvec[i] - 1))
            else:
                actions.append(self.get_next_natural_movement())
        return super().step(actions)

    def get_reward(self, old_states, new_states):
        reward_pickup = max(0, new_states["robot0_carrying"] - old_states["robot0_carrying"])
        reward_dropoff = 2 * (new_states["stall0_carrying"] - old_states["stall0_carrying"])
        time_penalty = 0.005
        return reward_pickup + reward_dropoff - time_penalty

    def get_done(self, states):
        return states["stall0_carrying"] == 2

    def init_natural_movement(self):
        self.current_action = random.randint(0, 5)
        self.time_till_switch = random.randint(5, 15)

    def get_next_natural_movement(self):
        self.time_till_switch -= 1
        if self.time_till_switch == 0:
            self.current_action = random.randint(0, 5)
            self.time_till_switch = random.randint(5, 15)
        return self.current_action


class VictimControlledBlocklandEnv(SingleAgentControlledBlocklandEnv):

    def __init__(self, level_id, adversary_policy=None, random_walk=False):
        with open(f"blockland/levels/{level_id}.json", "r") as file:
            level = json.load(file)
        super().__init__(level, "robot0", {"human0": adversary_policy}, random_walk)

    def should_include_state(self, state_id):
        excluded_states = ["robot0_z", "human0_z", "stall0_z", "box0_z", "box1_z"]
        return state_id not in excluded_states


class AdversaryControlledBlocklandEnv(SingleAgentControlledBlocklandEnv):

    def __init__(self, level_id, victim_policy=None, random_walk=False):
        with open(f"blockland/levels/{level_id}.json", "r") as file:
            level = json.load(file)
        super().__init__(level, "human0", {"robot0": victim_policy}, random_walk)

    def should_include_state(self, state_id):
        excluded_states = ["robot0_z", "human0_z", "stall0_z", "box0_z", "box1_z"]
        return state_id not in excluded_states

    def get_reward(self, old_states, new_states):
        return -super().get_reward(old_states, new_states)
