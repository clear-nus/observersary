import gym
import gym.spaces
import numpy as np

from evilslime import Color
from evilslime import EvilSlimeEnv


class VictimControlledEvilSlimeWrapper(gym.Wrapper):

    def __init__(self, env, background_color, adversary_policy=None):
        """Add a fixed adversarial slime to an environment.

        Args:
            env: Environment that is compliant with the OpenAI Gym API.
            background_color: Background color for EvilSlimeEnv.
            adversary_policy: Function that outputs an action given an observation. Random if None.
        """
        super().__init__(env)

        self.adversary_policy = adversary_policy

        self.victim_env = env
        adversary_color = Color.BLUE if self.adversary_policy is None else Color.RED
        self.adversary_env = EvilSlimeEnv(adversary_color, background_color)

        self.metadata["render_modes"] = ["rgb_array"]
        self.observation_space = gym.spaces.Box(
            np.hstack((self.victim_env.observation_space.low, self.adversary_env.observation_space.low)),
            np.hstack((self.victim_env.observation_space.high, self.adversary_env.observation_space.high)),
        )
        self.action_space = self.victim_env.action_space

    def step(self, victim_action):
        """Take a step through the environment.

        Args:
            victim_action: Victim action.
        """
        if self.adversary_policy is None:
            adversary_action = self.adversary_env.action_space.sample()
        elif self.adversary_policy is not None:
            adversary_action, _ = self.adversary_policy.predict(self.curr_obs)

        victim_obs, victim_reward, victim_done, victim_info = self.victim_env.step(victim_action)
        adversary_obs, _, _, _ = self.adversary_env.step(adversary_action)
        obs = np.append(victim_obs, adversary_obs)
        self.curr_obs = obs
        return obs, victim_reward, victim_done, victim_info

    def reset(self):
        """Reset the environment.

        """
        victim_obs = self.victim_env.reset()
        adversary_obs = self.adversary_env.reset()
        obs = np.append(victim_obs, adversary_obs)
        self.curr_obs = obs
        return obs

    def render(self, mode="rgb_array"):
        """Render the current state of the environment.

        Args:
            mode: Render mode. Only supports rgb_array.
        """
        victim_image = self.victim_env.render(mode=mode)
        adversary_image = self.adversary_env.render(mode=mode, render_width=victim_image.shape[1])
        return np.vstack((adversary_image, victim_image))

    def seed(self, seed):
        """Set the seed for random number generators.

        Args:
            seed: Seed value for the generators.
        """
        self.victim_env.seed(seed)
        self.adversary_env.seed(seed)


class AdversaryControlledEvilSlimeWrapper(gym.Wrapper):

    def __init__(self, env, background_color, victim_policy):
        """Add a controllable slime to an environment.

        Args:
            env: Environment that is compliant with the OpenAI Gym API.
            background_color: Background color for EvilSlimeEnv.
            victim_policy: Function that outputs an action given an observation. Random if None.
        """
        super().__init__(env)

        self.victim_policy = victim_policy

        self.victim_env = env
        self.adversary_env = EvilSlimeEnv(Color.YELLOW, background_color)

        self.metadata["render_modes"] = ["rgb_array"]
        self.observation_space = gym.spaces.Box(
            np.hstack((self.victim_env.observation_space.low, self.adversary_env.observation_space.low)),
            np.hstack((self.victim_env.observation_space.high, self.adversary_env.observation_space.high)),
        )
        self.action_space = self.adversary_env.action_space

    def step(self, adversary_action):
        """Take a step through the environment.

        Args:
            adversary_action: Adversary action.
        """
        if self.victim_policy is None:
            victim_action = self.victim_env.action_space.sample()
        elif self.victim_policy is not None:
            victim_action, _ = self.victim_policy.predict(self.curr_obs)

        victim_obs, victim_reward, victim_done, victim_info = self.victim_env.step(victim_action)
        adversary_obs, _, _, _ = self.adversary_env.step(adversary_action)
        obs = np.append(victim_obs, adversary_obs)
        self.curr_obs = obs
        return obs, -victim_reward, victim_done, victim_info

    def reset(self):
        """Reset the environment.

        """
        victim_obs = self.victim_env.reset()
        adversary_obs = self.adversary_env.reset()
        obs = np.append(victim_obs, adversary_obs)
        self.curr_obs = obs
        return obs

    def render(self, mode="rgb_array"):
        """Render the current state of the environment.

        Args:
            mode: Render mode. Only supports rgb_array.
        """
        victim_image = self.victim_env.render(mode=mode)
        adversary_image = self.adversary_env.render(mode=mode, render_width=victim_image.shape[1])
        return np.vstack((adversary_image, victim_image))

    def seed(self, seed):
        """Set the seed for random number generators.

        Args:
            seed: Seed value for the generators.
        """
        self.victim_env.seed(seed)
        self.adversary_env.seed(seed)
