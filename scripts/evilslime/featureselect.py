import argparse
import json

import xvfbwrapper

from observersary.experiments.evilslime import EvilSlimeEvaluationWithBlindExperiment
from observersary.experiments.evilslime import EvilSlimeFeatureSelectionExperiment


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--argfile", "-a", type=str, required=True)
    parser.add_argument("--exp-type", "-t", type=str, default="base", choices=["base", "l1", "noisy"])
    parser.add_argument("--remove-n", "-r", type=int, default=1)
    pargs = parser.parse_args()

    with open(pargs.argfile, "r") as argfile:
        args = json.load(argfile)

    base_name = f"evilslime/{args['env_id'].lower()}/{pargs.exp_type}"
    args["results_path"] = f"results/{base_name}/featureselect/results.csv"
    args["remove_n"] = pargs.remove_n

    if args["total_timesteps"] == 100000:
        checkpoints = [100]
    elif args["total_timesteps"] == 1000000:
        checkpoints = [1000]
    elif args["total_timesteps"] == 5000000:
        checkpoints = [5000]

    with xvfbwrapper.Xvfb():
        for iv in range(1, 21):
            for cv in checkpoints:
                exp_name = f"{base_name}/featureselect/v{iv:02d}-c{cv:04d}"
                victim_model = f"results/{base_name}/train/v{iv:02d}/models/checkpoint_{cv * 1000}_steps.zip"
                print(f"Running {exp_name}...")
                EvilSlimeFeatureSelectionExperiment({"exp_name": exp_name, "victim_model": victim_model, **args}).run()
                featureselect_path = f"results/{base_name}/featureselect/v{iv:02d}-c{cv:04d}/variances.csv"
                exp_name = f"{base_name}/featureselect/v{iv:02d}-c{cv:04d}/arand"
                victim_model = f"results/{base_name}/train/v{iv:02d}/models/checkpoint_{cv * 1000}_steps.zip"
                print(f"Running {exp_name}...")
                EvilSlimeEvaluationWithBlindExperiment({
                    "exp_name": exp_name,
                    "victim_model": victim_model,
                    "featureselect_path": featureselect_path,
                    **args
                }).run()
                exp_name = f"{base_name}/featureselect/v{iv:02d}-c{cv:04d}/aleft"
                victim_model = f"results/{base_name}/train/v{iv:02d}/models/checkpoint_{cv * 1000}_steps.zip"
                print(f"Running {exp_name}...")
                EvilSlimeEvaluationWithBlindExperiment({
                    "exp_name": exp_name,
                    "victim_model": victim_model,
                    "adversary_model": "left",
                    "featureselect_path": featureselect_path,
                    **args
                }).run()
                exp_name = f"{base_name}/featureselect/v{iv:02d}-c{cv:04d}/aright"
                victim_model = f"results/{base_name}/train/v{iv:02d}/models/checkpoint_{cv * 1000}_steps.zip"
                print(f"Running {exp_name}...")
                EvilSlimeEvaluationWithBlindExperiment({
                    "exp_name": exp_name,
                    "victim_model": victim_model,
                    "adversary_model": "right",
                    "featureselect_path": featureselect_path,
                    **args
                }).run()


if __name__ == "__main__":

    main()
