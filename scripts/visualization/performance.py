import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

task_path = sys.argv[1]
techniques = ["base", "l1", "noisy_0.01", "noisy_0.02", "noisy_0.05"]

results = []
datamin = 1e10
datamax = -1e10
xmin = datamin - (datamax - datamin) * 0.05
xmax = datamax + (datamax - datamin) * 0.05

for technique in techniques:
    results.append(pd.read_csv(f"{task_path}/{technique}/evaluate/results.csv", header=None))
    datamin = min(datamin, np.min(results[-1].loc[:, 1]))
    datamax = max(datamax, np.max(results[-1].loc[:, 2]))
    xmin = datamin - (datamax - datamin) * 0.05
    xmax = datamax + (datamax - datamin) * 0.05


def get_data(t):
    for i in range(20):
        indices = [(i * 9) + k + 6 for k in range(3)]
        adversary_label = list(map(lambda s: s.split("/")[5], results[t].loc[indices, 0].to_list()))
        info = results[t].loc[indices[0], 0].split("/")
        env_label = f"{info[1]} ({info[2]})"
        victim_label = info[4]
        data = results[t].loc[indices, 1:4].to_numpy()
        mins = data[:, 0]
        maxes = data[:, 1]
        means = data[:, 2]
        stds = data[:, 3]
        yield i, adversary_label, victim_label, env_label, mins, maxes, means, stds


fig, axs = plt.subplots(20, 5, figsize=(20, 20), sharex=True)
fig.set_facecolor("white")
for t, technique in enumerate(techniques):
    for i, adversary_label, victim_label, env_label, mins, maxes, means, stds in get_data(t):
        axs[i][t].errorbar(means[0], [0],
                           xerr=stds[0],
                           ecolor="#118AB233",
                           elinewidth=15,
                           lw=0,
                           marker="s",
                           markerfacecolor="#118AB2",
                           markeredgecolor="#118AB2")
        axs[i][t].errorbar(means[1:], [1, 2],
                           xerr=stds[1:],
                           ecolor="#EF476F33",
                           elinewidth=15,
                           lw=0,
                           marker="s",
                           markerfacecolor="#EF476F",
                           markeredgecolor="#EF476F")
        axs[i][t].errorbar(means[0], [0], xerr=[[means[0] - mins[0]], [maxes[0] - means[0]]], ecolor="#118AB2", elinewidth=2, lw=0)
        axs[i][t].errorbar(means[1:], [1, 2], xerr=[means[1:] - mins[1:], maxes[1:] - means[1:]], ecolor="#EF476F", elinewidth=2, lw=0)
        axs[i][t].set_xlim(xmin, xmax)
        # axs[i][t].set_xlim(-1500, 0)
        axs[i][t].set_ylim(-0.5, 2.5)
        axs[i][t].set_yticks([0, 1, 2])
        axs[i][t].set_yticklabels(adversary_label)
        axs[i][t].grid(axis="y")
        axs[i][t].set_title(f"{victim_label} ({technique})")
        axs[i][t].invert_yaxis()

fig.tight_layout()
fig.savefig(f"{task_path}/performance.pdf")
