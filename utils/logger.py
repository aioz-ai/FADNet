import os
import torch
import json


class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir

    def write_model(self, model_params, iteration=0, mode="json"):
        """
        save model parameters as .pt file
        :param model_params: torch.tensor
        :param iteration: integer
        :param mode:
        """
        if mode == "torch":
            file_path = os.path.join(self.logdir,
                                     "model_{}.pt".format(iteration))
            torch.save(model_params, file_path)

        elif mode == "json":
            file_path = os.path.join(self.logdir,
                                     "model_{}.json".format(iteration))

            with open(file_path, "w") as f:
                f.write(json.dumps(model_params.tolist()))