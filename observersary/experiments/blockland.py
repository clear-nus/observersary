import os

import gym
import gym.spaces
import numpy as np
import stable_baselines3 as sb3
import stable_baselines3.common.monitor as sb3monitor
import stable_baselines3.common.vec_env as sb3vec

from observersary.environments.blockland import AdversaryControlledBlocklandEnv
from observersary.environments.blockland import VictimControlledBlocklandEnv
from observersary.experiments.base import DeterministicExperiment
from observersary.experiments.training import AbstractTrainingExperiment


class BlocklandVictimTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.level_id = self.add_argument("level_id", None)
        self.random_walk = self.add_argument("random_walk", False)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = VictimControlledBlocklandEnv(self.level_id, random_walk=self.random_walk)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class BlocklandAdversaryTrainingExperiment(AbstractTrainingExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.level_id = self.add_argument("level_id", None)
        self.victim_model = self.add_argument("victim_model", None)
        self.random_walk = self.add_argument("random_walk", False)

    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)
        episode_trigger = lambda episode: episode % 5 == 0

        def make_env_fn(idx):

            def make_env():

                env = AdversaryControlledBlocklandEnv(self.level_id, victim_policy, random_walk=self.random_walk)
                env = sb3monitor.Monitor(env)
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"{self.root_path}/videos", episode_trigger)
                env.seed(self.seed + idx)
                env.action_space.seed(self.seed + idx)
                env.observation_space.seed(self.seed + idx)
                return env

            return make_env

        return sb3vec.SubprocVecEnv([make_env_fn(idx) for idx in range(self.n_envs)])


class BlocklandEvaluationExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.level_id = self.add_argument("level_id", None)
        self.num_evaluations = self.add_argument("num_updates", 30)
        self.victim_model = self.add_argument("victim_model", None)
        self.adversary_model = self.add_argument("adversary_model", None)
        self.random_walk = self.add_argument("random_walk", False)
        self.results_path = self.add_argument("results_path", None)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        victim_policy = sb3.PPO.load(self.victim_model, device=self.device)

        adversary_policy = None
        if self.adversary_model is not None:
            adversary_policy = sb3.PPO.load(self.adversary_model, device=self.device)

        episode_trigger = lambda episode: episode == 0

        def make_env_fn(idx):

            def make_env():

                env = VictimControlledBlocklandEnv(self.level_id, adversary_policy, self.random_walk)
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
