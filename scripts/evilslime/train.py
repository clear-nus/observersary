import argparse
import json

import xvfbwrapper

from observersary.experiments.evilslime import EvilSlimeVictimL1RegularizedTrainingExperiment
from observersary.experiments.evilslime import EvilSlimeVictimNoisyTrainingExperiment
from observersary.experiments.evilslime import EvilSlimeVictimTrainingExperiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--argfile", "-a", type=str, required=True)
    parser.add_argument("--exp-type", "-t", type=str, default="base", choices=["base", "l1", "noisy"])
    pargs = parser.parse_args()

    with open(pargs.argfile, "r") as argfile:
        args = json.load(argfile)

    if pargs.exp_type == "base":
        VictimTrainingExperiment = EvilSlimeVictimTrainingExperiment
    elif pargs.exp_type == "l1":
        VictimTrainingExperiment = EvilSlimeVictimL1RegularizedTrainingExperiment
    elif pargs.exp_type == "noisy":
        VictimTrainingExperiment = EvilSlimeVictimNoisyTrainingExperiment

    base_name = f"evilslime/{args['env_id'].lower()}/{pargs.exp_type}"

    with xvfbwrapper.Xvfb():
        for i in range(1, 21):
            exp_name = f"{base_name}/train/v{i:02d}"
            print(f"Running {exp_name}...")
            VictimTrainingExperiment({"exp_name": exp_name, **args}).run()


if __name__ == "__main__":

    main()
