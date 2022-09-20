import datetime
import json
import os
import random
import time

import numpy as np
import torch


class BaseExperiment:

    def __init__(self, argstore):
        """Base class for an experiment.

        Args:
            argstore: Dictionary of potentially used arguments.
        """
        self.argstore = argstore
        self.args = {}

    def add_argument(self, name, default):
        """Add an argument.

        Args:
            name: Name of argument.
            default: Default value for argument. To be used when name is not found in argstore.
        """
        self.args[name] = default
        if name in self.argstore:
            self.args[name] = self.argstore[name]
        return self.args[name]

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        self.exp_name = self.add_argument("exp_name", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def run(self):
        """Run the experiment.

        """
        self.parse_arguments()
        self.root_path = f"results/{self.exp_name}"
        os.makedirs(self.root_path, exist_ok=True)
        with open(f"{self.root_path}/arguments.json", "w") as file:
            json.dump(self.args, file, indent=4)


class DeterministicExperiment(BaseExperiment):

    def parse_arguments(self):
        """Parse all arguments. To be executed before run().

        """
        super().parse_arguments()

        self.seed = self.add_argument("seed", int(time.time()))
        self.cuda = self.add_argument("cuda", False)

    def run(self):
        """Run the experiment with determinism.

        """
        super().run()

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
