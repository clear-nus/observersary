import argparse
import json

import xvfbwrapper

from observersary.experiments.blockland import BlocklandAdversaryTrainingExperiment
from observersary.experiments.blockland import BlocklandVictimTrainingExperiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--argfile", "-a", type=str, required=True)
    parser.add_argument("--exp-type", "-t", type=str, default="base", choices=["base"])
    pargs = parser.parse_args()

    with open(pargs.argfile, "r") as argfile:
        args = json.load(argfile)

    if pargs.exp_type == "base":
        VictimTrainingExperiment = BlocklandVictimTrainingExperiment
        AdversaryTrainingExperiment = BlocklandAdversaryTrainingExperiment

    base_name = f"blockland/{args['level_id'].lower()}/{pargs.exp_type}"

    with xvfbwrapper.Xvfb():
        for iv in range(4, 6):
            exp_name = f"{base_name}/train/v{iv:02d}"
            print(f"Running {exp_name}...")
            VictimTrainingExperiment({"exp_name": exp_name, **args}).run()
            for ia in range(1, 4):
                exp_name = f"{base_name}/train/v{iv:02d}-a{ia:02d}"
                victim_model = f"results/{base_name}/train/v{iv:02d}/models/best_model.zip"
                print(f"Running {exp_name}...")
                AdversaryTrainingExperiment({"exp_name": exp_name, "victim_model": victim_model, **args}).run()


if __name__ == "__main__":

    main()
