"""
Class which holds config file for optimizer. The optimizer
is always SGD, but in the training.py it can be adjusted.
"""

import json
import os

class OptimizerConfig(object):
    '''
    Class for specifying OptimizerConfig
    '''
    
    '''
    optimizer_name: string, name of the optimizer which is implemented
    in the training.py script
    '''
    optimizer_name = "SGD"

    '''
    learning_rate: float, initial learning rate of the optimizer
    '''
    learning_rate = 1e-2

    '''
    weight_decay: float, weight decay of optimizer
    '''
    weight_decay = 3e-5

    '''
    momentum: float, momentum of the optimizer
    '''
    momentum = 0.99

    '''
    nesterov: boolean, if set to True, neseterov momenum will be 
    enabled
    '''
    nesterov = True

    '''
    scheduler_name: string, name of the scheduler. It can be either 
    "poly" or "fibo"
    '''
    scheduler_name = "poly"

    '''
    arch_p: float, in case we use "fibo" scheduler, this is additional 
    paramter which is a requirement by the scheduler
    '''
    arch_p = 0.2

    def __str__(self):
        """
        Return a formatted string of all optimizer configuration parameters.
        """
        _retval = ["######################################",
                 "Optimizer Configuration:"]
        _retval.append(f"optimizer_name: {self.optimizer_name}")
        _retval.append(f"learning_rate: {self.learning_rate}")
        _retval.append(f"weight_decay: {self.weight_decay}")
        _retval.append(f"momentum: {self.momentum}")
        _retval.append(f"nesterov: {self.nesterov}")
        _retval.append(f"scheduler_name: {self.scheduler_name}")
        _retval.append(f"arch_p: {self.arch_p}")
        _retval.append("######################################")
        return "\n".join(_retval)

    def export_to_json(self, 
                       export_path: str = None):
        """
        Export this optimizer config to a JSON file.
        If export_path is a directory, writes to <export_path>/optimizer_config.json.

        Args:
            export_path (str): Path to directory or full file path for JSON export.
        """
        _data = {
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "scheduler_name": self.scheduler_name,
            "arch_p": self.arch_p
        }

        if export_path is None:
            print("export_path is None. Config cannot be exported!")
            return

        # If given a directory, append a filename
        if not export_path.endswith(".json"):
            if not os.path.isdir(export_path):
                raise ValueError(f"Directory does not exist: {export_path}")
            export_path = os.path.join(export_path, "optimizer_config.json")

        with open(export_path, "w") as _f:
            json.dump(_data, _f, indent=4)
        print(f"Optimizer configuration exported to {export_path}")

    def import_from_json(self, 
                         import_path: str):
        """
        Load optimizer config from a JSON file previously written by export_to_json.

        Args:
            import_path (str): Path to the .json file exported by this class.
        """
        if not import_path.endswith(".json") or not os.path.isfile(import_path):
            raise FileNotFoundError(f"JSON file not found: {import_path}")

        with open(import_path, "r") as _f:
            _data = json.load(_f)

        # Assign loaded values to this instance
        self.optimizer_name = _data.get("optimizer_name", self.optimizer_name)
        self.learning_rate = _data.get("learning_rate", self.learning_rate)
        self.weight_decay = _data.get("weight_decay", self.weight_decay)
        self.momentum = _data.get("momentum", self.momentum)
        self.nesterov = _data.get("nesterov", self.nesterov)
        self.scheduler_name = _data.get("scheduler_name", self.scheduler_name)
        self.arch_p = _data.get("arch_p", self.arch_p)
        print(f"Optimizer configuration imported from {import_path}")
