import argparse
import json

import xvfbwrapper

from observersary.experiments.blockland import BlocklandEvaluationExperiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--argfile", "-a", type=str, required=True)
    parser.add_argument("--exp-type", "-t", type=str, default="base", choices=["base"])
    pargs = parser.parse_args()

    with open(pargs.argfile, "r") as argfile:
        args = json.load(argfile)

    base_name = f"blockland/{args['level_id'].lower()}/{pargs.exp_type}"
    args["results_path"] = f"results/{base_name}/evaluate/results.csv"

    with xvfbwrapper.Xvfb():
        for iv in range(1, 6):
            exp_name = f"{base_name}/evaluate/v{iv:02d}/arand"
            victim_model = f"results/{base_name}/train/v{iv:02d}/models/best_model.zip"
            print(f"Running {exp_name}...")
            BlocklandEvaluationExperiment({"exp_name": exp_name, "victim_model": victim_model, **args}).run()
            for iva in range(1, 6):
                for ia in range(1, 4):
                    exp_name = f"{base_name}/evaluate/v{iv:02d}/v{iva:02d}-a{ia:02d}"
                    victim_model = f"results/{base_name}/train/v{iv:02d}/models/best_model.zip"
                    adversary_model = f"results/{base_name}/train/v{iva:02d}-a{ia:02d}/models/best_model.zip"
                    print(f"Running {exp_name}...")
                    BlocklandEvaluationExperiment({"exp_name": exp_name, "victim_model": victim_model, "adversary_model": adversary_model, **args}).run()


if __name__ == "__main__":
    main()
