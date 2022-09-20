import abc

import stable_baselines3.common.callbacks as sb3callbacks
import stable_baselines3.common.logger as sb3logger

from observersary.algorithms.ppo import PPO
from observersary.algorithms.ppo import PPOWithInputL1Regularizer
from observersary.experiments.base import DeterministicExperiment


class AbstractTrainingExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.policy = self.add_argument("policy", "MlpPolicy")
        self.learning_rate = self.add_argument("learning_rate", 0.0003)
        self.n_steps = self.add_argument("n_steps", 2048)
        self.batch_size = self.add_argument("batch_size", 64)
        self.n_epochs = self.add_argument("n_epochs", 10)
        self.gamma = self.add_argument("gamma", 0.99)
        self.gae_lambda = self.add_argument("gae_lambda", 0.95)
        self.clip_range = self.add_argument("clip_range", 0.2)
        self.ent_coef = self.add_argument("ent_coef", 0.0)
        self.vf_coef = self.add_argument("vf_coef", 0.5)
        self.max_grad_norm = self.add_argument("max_grad_norm", 0.5)
        self.use_sde = self.add_argument("use_sde", False)
        self.sde_sample_freq = self.add_argument("sde_sample_freq", -1)

        self.n_eval_episodes = self.add_argument("n_eval_episodes", 10)
        self.eval_freq = self.add_argument("eval_freq", 625)
        self.total_timesteps = self.add_argument("total_timesteps", 100000)
        self.n_envs = self.add_argument("n_envs", 4)
        self.n_checkpoints = self.add_argument("n_checkpoints", 20)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        train_envs = self.get_envs()
        eval_envs = self.get_envs()

        anneal_lr = lambda factor: factor * self.learning_rate
        model = PPO(policy=self.policy,
                    env=train_envs,
                    learning_rate=anneal_lr,
                    n_steps=self.n_steps,
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    clip_range=self.clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    max_grad_norm=self.max_grad_norm,
                    verbose=0,
                    seed=self.seed,
                    device=self.device)

        logger = sb3logger.configure(f"{self.root_path}/logs", ["csv", "stdout", "tensorboard"])
        model.set_logger(logger)

        save_freq = (self.total_timesteps // self.n_checkpoints) // self.n_envs
        checkpoint_callback = sb3callbacks.CheckpointCallback(save_freq=save_freq, save_path=f"{self.root_path}/models", name_prefix="checkpoint")

        eval_callback = sb3callbacks.EvalCallback(eval_env=eval_envs,
                                                  n_eval_episodes=self.n_eval_episodes,
                                                  eval_freq=self.eval_freq,
                                                  best_model_save_path=f"{self.root_path}/models",
                                                  deterministic=True)

        callbacks = sb3callbacks.CallbackList([checkpoint_callback, eval_callback])
        model.learn(total_timesteps=self.total_timesteps, callback=callbacks)

        train_envs.close()
        eval_envs.close()

    @abc.abstractmethod
    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        raise NotImplementedError


class AbstractL1RegularizedTrainingExperiment(DeterministicExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.policy = self.add_argument("policy", "MlpPolicy")
        self.learning_rate = self.add_argument("learning_rate", 0.0003)
        self.n_steps = self.add_argument("n_steps", 2048)
        self.batch_size = self.add_argument("batch_size", 64)
        self.n_epochs = self.add_argument("n_epochs", 10)
        self.gamma = self.add_argument("gamma", 0.99)
        self.gae_lambda = self.add_argument("gae_lambda", 0.95)
        self.clip_range = self.add_argument("clip_range", 0.2)
        self.ent_coef = self.add_argument("ent_coef", 0.0)
        self.vf_coef = self.add_argument("vf_coef", 0.5)
        self.max_grad_norm = self.add_argument("max_grad_norm", 0.5)
        self.use_sde = self.add_argument("use_sde", False)
        self.sde_sample_freq = self.add_argument("sde_sample_freq", -1)

        self.n_eval_episodes = self.add_argument("n_eval_episodes", 10)
        self.eval_freq = self.add_argument("eval_freq", 625)
        self.total_timesteps = self.add_argument("total_timesteps", 100000)
        self.n_envs = self.add_argument("n_envs", 4)
        self.n_checkpoints = self.add_argument("n_checkpoints", 20)

    def run(self):
        """Run the training experiment.

        """
        super().run()

        train_envs = self.get_envs()
        eval_envs = self.get_envs()

        anneal_lr = lambda factor: factor * self.learning_rate
        model = PPOWithInputL1Regularizer(policy=self.policy,
                                          env=train_envs,
                                          learning_rate=anneal_lr,
                                          n_steps=self.n_steps,
                                          batch_size=self.batch_size,
                                          n_epochs=self.n_epochs,
                                          gamma=self.gamma,
                                          gae_lambda=self.gae_lambda,
                                          clip_range=self.clip_range,
                                          ent_coef=self.ent_coef,
                                          vf_coef=self.vf_coef,
                                          max_grad_norm=self.max_grad_norm,
                                          verbose=0,
                                          seed=self.seed,
                                          device=self.device)

        logger = sb3logger.configure(f"{self.root_path}/logs", ["csv", "stdout", "tensorboard"])
        model.set_logger(logger)

        save_freq = (self.total_timesteps // self.n_checkpoints) // self.n_envs
        checkpoint_callback = sb3callbacks.CheckpointCallback(save_freq=save_freq, save_path=f"{self.root_path}/models", name_prefix="checkpoint")

        eval_callback = sb3callbacks.EvalCallback(eval_env=eval_envs,
                                                  n_eval_episodes=self.n_eval_episodes,
                                                  eval_freq=self.eval_freq,
                                                  best_model_save_path=f"{self.root_path}/models",
                                                  deterministic=True)

        callbacks = sb3callbacks.CallbackList([checkpoint_callback, eval_callback])
        model.learn(total_timesteps=self.total_timesteps, callback=callbacks)

        train_envs.close()
        eval_envs.close()

    @abc.abstractmethod
    def get_envs(self):
        """Initialize a vectorized environment used for training and evaluation.

        """
        raise NotImplementedError
