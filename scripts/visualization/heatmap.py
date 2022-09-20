import sys

import matplotlib.pyplot as plt
import numpy as np
import xvfbwrapper

from observersary.environments.blockland import VictimControlledBlocklandEnv

level_id = sys.argv[1]
random_walk = level_id[-3:] == "-rw"

with xvfbwrapper.Xvfb():
    env = VictimControlledBlocklandEnv(level_id, random_walk=random_walk)
    X = np.zeros((len(env.floors[0]) * 5, len(env.floors) * 5))
    for i in range(1000):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, info = env.step(0)
            X[int(obs[4] * 5), int(obs[3] * 5)] += 1

np.save(f"heatmap-{level_id}.npy", X)
fig, ax = plt.subplots()
ax.imshow(X, cmap=plt.get_cmap("YlOrRd_r"), vmin=0, vmax=100, aspect="auto")
ax.invert_yaxis()
fig.savefig(f"heatmap-{level_id}.pdf")
