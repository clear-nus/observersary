import os

import gym
import gym.spaces
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import stable_baselines3.common.monitor as sb3monitor
import stable_baselines3.common.vec_env as sb3vec
import torch

from evilslime import Color
from observersary.environments.commons import BlindObservationWrapper
from observersary.environments.commons import NoisyObservationWrapper
from observersary.environments.evilslime import AdversaryControlledEvilSlimeWrapper
from observersary.environments.evilslime import VictimControlledEvilSlimeWrapper
from observersary.experiments.base import DeterministicExperiment
from observersary.experiments.training import AbstractL1RegularizedTrainingExperiment
from observersary.experiments.training import AbstractTrainingExperiment


class EvilSlimeVictimTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = VictimControlledEvilSlimeWrapper(env, self.background_color)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class EvilSlimeAdversaryTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.victim_model = self.add_argument("victim_model", None)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = AdversaryControlledEvilSlimeWrapper(env, self.background_color, victim_policy)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class EvilSlimeVictimL1RegularizedTrainingExperiment(AbstractL1RegularizedTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = VictimControlledEvilSlimeWrapper(env, self.background_color)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class EvilSlimeAdversaryL1RegularizedTrainingExperiment(AbstractL1RegularizedTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.victim_model = self.add_argument("victim_model", None)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = AdversaryControlledEvilSlimeWrapper(env, self.background_color, victim_policy)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class EvilSlimeVictimNoisyTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.noise_rate = self.add_argument("noise_rate", 0.05)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = VictimControlledEvilSlimeWrapper(env, self.background_color)
                env = NoisyObservationWrapper(env, self.noise_rate)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class EvilSlimeAdversaryNoisyTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", "CartPole-v1")
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.victim_model = self.add_argument("victim_model", None)
        self.noise_rate = self.add_argument("noise_rate", 0.05)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = AdversaryControlledEvilSlimeWrapper(env, self.background_color, victim_policy)
                env = NoisyObservationWrapper(env, self.noise_rate)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class ConstantActor:

    def __init__(self, action):

        self.action = action

    def predict(self, _):

        return (self.action, None)


class EvilSlimeEvaluationExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", None)
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.num_evaluations = self.add_argument("num_updates", 30)
        self.victim_model = self.add_argument("victim_model", None)
        self.adversary_model = self.add_argument("adversary_model", None)
        self.results_path = self.add_argument("results_path", None)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)

        adversary_policy = None
        if self.adversary_model == "left":
            adversary_policy = ConstantActor(0)
        elif self.adversary_model == "right":
            adversary_policy = ConstantActor(2)
        elif self.adversary_model is not None:
            adversary_policy = sb3.PPO.load(self.adversary_model, device=self.device)

        episode_trigger = lambda episode: episode == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = VictimControlledEvilSlimeWrapper(env, self.background_color, adversary_policy)
                env = sb3monitor.Monitor(env)
                env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger, name_prefix=f"{idx:03d}")
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        envs = sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.num_evaluations)])

        returns = np.zeros((self.num_evaluations))
        obs = envs.reset()
        done_once = np.array([False] * self.num_evaluations)
        while not np.all(done_once):
            victim_action, _ = victim_policy.predict(obs)
            obs, reward, done, _ = envs.step(victim_action)
            returns += reward * (1 - done_once.astype(int))
            done_once = np.logical_or(done_once, done)

        with open(self.results_path, "a") as file:
            file.write(f"{self.exp_name},{returns.min()},{returns.max()},{returns.mean()},{returns.std()},")
            file.write(f"{','.join(map(str, returns))}\n")
        print(f"{returns.min()  = }\n{returns.max()  = }\n{returns.mean() = }\n{returns.std()  = }")

        envs.close()

        videos_path = f"{self.root_path}/videos"
        for item in os.listdir(videos_path):
            if item.endswith(".json"):
                os.remove(os.path.join(videos_path, item))
            if item.endswith(".mp4"):
                os.rename(os.path.join(videos_path, item), os.path.join(videos_path, f"rl-video-episode-{int(item[:3])}.mp4"))


class EvilSlimeFeatureSelectionExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", None)
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.victim_model = self.add_argument("victim_model", None)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)
        adversary_policy = None

        env = gym.make(self.env_id)
        env = VictimControlledEvilSlimeWrapper(env, self.background_color, adversary_policy)
        env = sb3monitor.Monitor(env)
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)

        num_samples = 3000
        obses = np.zeros((num_samples, env.observation_space.shape[0]))
        curr_samples = 0

        while curr_samples < num_samples:
            obs = env.reset()
            if np.random.uniform() < 0.2:
                obses[curr_samples, :] = obs
                curr_samples += 1
            done = False
            while not done and curr_samples < num_samples:
                obs, _, done, _ = env.step(env.action_space.sample())
                if np.random.uniform() < 0.2:
                    obses[curr_samples, :] = obs
                    curr_samples += 1

        variances = np.zeros((num_samples, env.observation_space.shape[0]))
        for i in range(num_samples):
            original_obs = obses[i, :].copy().reshape(-1, 1).repeat(num_samples, 1).T
            for dim in range(env.observation_space.shape[0]):
                obs = original_obs.copy()
                obs[:, dim] = obses[:, dim]
                values = victim_policy.policy.predict_values(torch.tensor(obs)).reshape((-1)).detach().numpy()
                variances[i, dim] = np.var(values)

        with open(f"{self.root_path}/variances.csv", "a") as file:
            file.write(f"{','.join(map(str, np.mean(variances, axis=0)))}\n")
            file.write(f"{','.join(map(str, np.mean(obs, axis=0)))}\n")

        env.close()


class EvilSlimeEvaluationWithBlindExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.env_id = self.add_argument("env_id", None)
        self.background_color = self.add_argument("background_color", Color.WHITE)
        self.num_evaluations = self.add_argument("num_updates", 30)
        self.victim_model = self.add_argument("victim_model", None)
        self.adversary_model = self.add_argument("adversary_model", None)
        self.featureselect_path = self.add_argument("featureselect_path", None)
        self.remove_n = self.add_argument("remove_n", 1)
        self.results_path = self.add_argument("results_path", None)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)

        adversary_policy = None
        if self.adversary_model == "left":
            adversary_policy = ConstantActor(0)
        elif self.adversary_model == "right":
            adversary_policy = ConstantActor(2)
        elif self.adversary_model is not None:
            adversary_policy = sb3.PPO.load(self.adversary_model, device=self.device)

        featureselect_results = np.genfromtxt(self.featureselect_path, delimiter=',')
        features_importance = featureselect_results[0, :].argsort()
        blind_defaults = featureselect_results[1, :]

        if self.remove_n == -1:
            irrelevant_index = np.where(features_importance == features_importance.shape[0] - 1)[0][0]
            blind_indices = features_importance[:irrelevant_index + 1]
        else:
            blind_indices = features_importance[:self.remove_n]

        episode_trigger = lambda episode: episode == 0

        def make_env_fn(idx):

            def make_env():

                env = gym.make(self.env_id)
                env = VictimControlledEvilSlimeWrapper(env, self.background_color, adversary_policy)
                env = BlindObservationWrapper(env, blind_indices, blind_defaults)
                env = sb3monitor.Monitor(env)
                env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger, name_prefix=f"{idx:03d}")
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        envs = sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.num_evaluations)])

        returns = np.zeros((self.num_evaluations))
        obs = envs.reset()
        done_once = np.array([False] * self.num_evaluations)
        while not np.all(done_once):
            victim_action, _ = victim_policy.predict(obs)
            obs, reward, done, _ = envs.step(victim_action)
            returns += reward * (1 - done_once.astype(int))
            done_once = np.logical_or(done_once, done)

        with open(self.results_path, "a") as file:
            file.write(f"{self.exp_name},{returns.min()},{returns.max()},{returns.mean()},{returns.std()},")
            file.write(f"{','.join(map(str, returns))}\n")
        print(f"{returns.min()  = }\n{returns.max()  = }\n{returns.mean() = }\n{returns.std()  = }")

        envs.close()

        videos_path = f"{self.root_path}/videos"
        for item in os.listdir(videos_path):
            if item.endswith(".json"):
                os.remove(os.path.join(videos_path, item))
            if item.endswith(".mp4"):
                os.rename(os.path.join(videos_path, item), os.path.join(videos_path, f"rl-video-episode-{int(item[:3])}.mp4"))
