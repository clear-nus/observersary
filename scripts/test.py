import sys

import gym
import gym.spaces
import numpy as np
import stable_baselines3 as sb3
import stable_baselines3.common.monitor as sb3monitor
import stable_baselines3.common.vec_env as sb3vec
import torch

from observersary.environments.evilslime import VictimControlledEvilSlimeWrapper

victim_model = sys.argv[1]
env_id = "LunarLander-v2"
background_color = "#ffffff"
seed = 0

victim_policy = sb3.PPO.load(victim_model, device="cpu")

adversary_policy = None

episode_trigger = lambda episode: episode == 0


def make_env_fn(idx):

    def make_env():

        env = gym.make(env_id)
        env = VictimControlledEvilSlimeWrapper(env, background_color, adversary_policy)
        env = sb3monitor.Monitor(env)
        env.seed(seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return make_env


env = make_env_fn(0)()
n_samples = 2000
obses = np.zeros((n_samples, env.observation_space.shape[0]))
t = 0

while t < n_samples:
    obs = env.reset()
    if np.random.uniform() < 0.5:
        obses[t, :] = obs
        t += 1
    done = False
    while not done and t < n_samples:
        victim_action, _ = victim_policy.predict(obs)
        obs, reward, done, _ = env.step(env.action_space.sample())
        if np.random.uniform() < 0.5:
            obses[t, :] = obs
            t += 1

variances = np.zeros((n_samples, env.observation_space.shape[0]))
for i in range(n_samples):
    original_obs = obses[i, :].copy().reshape(-1, 1).repeat(n_samples, 1).T
    for dim in range(env.observation_space.shape[0]):
        obs = original_obs.copy()
        obs[:, dim] = obses[:, dim]
        values = victim_policy.policy.predict_values(torch.tensor(obs)).reshape((-1)).detach().numpy()
        variances[i, dim] = np.var(values)
print(",".join(np.mean(variances, axis=0)))

env.close()
