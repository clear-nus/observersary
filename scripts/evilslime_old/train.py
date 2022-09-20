import argparse
import json

import xvfbwrapper

from observersary.experiments.evilslime import EvilSlimeAdversaryL1RegularizedTrainingExperiment
from observersary.experiments.evilslime import EvilSlimeAdversaryTrainingExperiment
from observersary.experiments.evilslime import EvilSlimeVictimL1RegularizedTrainingExperiment
from observersary.experiments.evilslime import EvilSlimeVictimTrainingExperiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--argfile", "-a", type=str, required=True)
    parser.add_argument("--exp-type", "-t", type=str, default="base", choices=["base", "l1"])
    pargs = parser.parse_args()

    with open(pargs.argfile, "r") as argfile:
        args = json.load(argfile)

    if pargs.exp_type == "base":
        VictimTrainingExperiment = EvilSlimeVictimTrainingExperiment
        AdversaryTrainingExperiment = EvilSlimeAdversaryTrainingExperiment
    elif pargs.exp_type == "l1":
        VictimTrainingExperiment = EvilSlimeVictimL1RegularizedTrainingExperiment
        AdversaryTrainingExperiment = EvilSlimeAdversaryL1RegularizedTrainingExperiment

    base_name = f"evilslime/{args['env_id'].lower()}/{pargs.exp_type}"

    if args["total_timesteps"] == 100000:
        checkpoints = [60, 80, 100]
    elif args["total_timesteps"] == 1000000:
        checkpoints = [600, 800, 1000]
    elif args["total_timesteps"] == 5000000:
        checkpoints = [3000, 4000, 5000]

    with xvfbwrapper.Xvfb():
        for i in range(1, 6):
            exp_name = f"{base_name}/train/v{i:02d}"
            print(f"Running {exp_name}...")
            VictimTrainingExperiment({"exp_name": exp_name, **args}).run()
            for c in checkpoints:
                exp_name = f"{base_name}/train/v{i:02d}-c{c:04d}-a01"
                victim_model = f"results/{base_name}/train/v{i:02d}/models/checkpoint_{c * 1000}_steps.zip"
                print(f"Running {exp_name}...")
                AdversaryTrainingExperiment({"exp_name": exp_name, "victim_model": victim_model, **args, "use_sde": False}).run()


if __name__ == "__main__":

    main()
