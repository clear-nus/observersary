import gym
import numpy as np


class NoisyObservationWrapper(gym.Wrapper):

    def __init__(self, env, noise_rate):
        """Add noise to the observations of an environment.

        Args:
            env: Environment that is compliant with the OpenAI Gym API.
            rate: Probability of noisy observation.
        """
        super().__init__(env)
        self.noise_rate = noise_rate

    def step(self, action):
        """Take a step through the environment.

        Args:
            action: Agent action.
        """
        obs, reward, done, info = self.env.step(action)
        random_obs = self.env.observation_space.sample()
        for i in range(len(obs)):
            if np.random.uniform() < self.noise_rate:
                obs[i] = random_obs[i]
        return obs, reward, done, info


class BlindObservationWrapper(gym.Wrapper):

    def __init__(self, env, blind_indices, blind_defaults):
        """Blind certain observations of an environment.

        Args:
            env: Environment that is compliant with the OpenAI Gym API.
            blind_indices: List of feature indices to be blinded.
            blind_defaults: Default values for blinded features.
        """
        super().__init__(env)
        self.blind_indices = blind_indices
        self.blind_defaults = blind_defaults

    def step(self, action):
        """Take a step through the environment.

        Args:
            action: Agent action.
        """
        obs, reward, done, info = self.env.step(action)
        obs[self.blind_indices] = self.blind_defaults[self.blind_indices]
        return obs, reward, done, info
