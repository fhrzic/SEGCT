"""
Class which holds config file for training process.
"""

import json
import os

class TrainingConfig(object):
    '''
    Class for specifying TrainingConfig
    '''

    '''
    epochs: int, number of epochs for training
    '''
    epochs = 1000

    '''
    batch_per_epoch: int, number of batches per epoch
    '''
    batch_per_epoch = 250

    '''
    batch_size: int, number of elements per batch
    '''
    batch_size = 4

    '''
    optimizer_params: instance of OptimizerConfig class or None
    '''
    optimizer_params = None

    '''
    accelerator: bool, use accelerate library (parallel computing)
    '''
    accelerator = False

    '''
    save_model_interval: int, after how many epochs model will be saved
    '''
    save_model_interval = 5

    '''
    training_loss_function: name or config of the loss function for training
    '''
    training_loss_function = None

    '''
    validation_loss_function: name or config of the loss function for validation
    '''
    validation_loss_function = None

    '''
    device: str, device to be used ('cpu' or 'cuda')
    '''
    device = "cpu"

    '''
    output_dir: str, directory where the models will be saved
    '''
    output_dir = "Trained_Models"

    '''
    resume_from_checkpoint: str or None, path to last checkpoint
    '''
    resume_from_checkpoint = None

    '''
    gradient_scaler: bool, whether to use gradient scaler
    '''
    gradient_scaler = False

    '''
    mixed_precision: bool, whether to use mixed precision
    '''
    mixed_precision = False

    def __str__(self):
        """
        Return a formatted string of all training configuration parameters.
        """
        _retval = ["######################################",
                   "Training Configuration:"]
        _retval.append(f"epochs: {self.epochs}")
        _retval.append(f"batch_per_epoch: {self.batch_per_epoch}")
        _retval.append(f"batch_size: {self.batch_size}")
        _retval.append(f"optimizer_params: {self.optimizer_params}")
        _retval.append(f"save_model_interval: {self.save_model_interval}")
        _retval.append(f"device: {self.device}")
        _retval.append(f"output_dir: {self.output_dir}")
        _retval.append(f"resume_from_checkpoint: {self.resume_from_checkpoint}")
        _retval.append(f"accelerator: {self.accelerator}")
        _retval.append(f"gradient_scaler: {self.gradient_scaler}")
        _retval.append(f"mixed_precision: {self.mixed_precision}")
        _retval.append(f"training_loss_function: {self.training_loss_function.__class__.__name__}")
        _retval.append(f"validation_loss_function: {self.validation_loss_function.__class__.__name__}")
        _retval.append("######################################")
        return "\n".join(_retval)

    def export_to_json(self, 
                       export_path: str = None):
        """
        Export this training config to a JSON file.
        If export_path is a directory, writes to <export_path>/training_config.json.

        Args:
            export_path (str): Path to directory or full file path for JSON export.
        """
        _data = {
            "epochs": self.epochs,
            "batch_per_epoch": self.batch_per_epoch,
            "batch_size": self.batch_size,
            "optimizer_params": self.optimizer_params.__class__.__name__,
            "save_model_interval": self.save_model_interval,
            "device": self.device,
            "output_dir": self.output_dir,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "accelerator": self.accelerator,
            "gradient_scaler": self.gradient_scaler,
            "mixed_precision": self.mixed_precision,
            "training_loss_function": self.training_loss_function.__class__.__name__,
            "validation_loss_function": self.validation_loss_function.__class__.__name__
        }

        if export_path is None:
            print("export_path is None. Config cannot be exported!")
            return

        if not export_path.endswith(".json"):
            if not os.path.isdir(export_path):
                raise ValueError(f"Directory does not exist: {export_path}")
            export_path = os.path.join(export_path, "training_config.json")

        with open(export_path, "w") as _f:
            json.dump(_data, _f, indent=4)
        print(f"Training configuration exported to {export_path}")

    def import_from_json(self, 
                         import_path: str):
        """
        Load training config from a JSON file previously written by export_to_json.

        Args:
            import_path (str): Path to the .json file exported by this class.
        """
        if not import_path.endswith(".json") or not os.path.isfile(import_path):
            raise FileNotFoundError(f"JSON file not found: {import_path}")

        with open(import_path, "r") as _f:
            _data = json.load(_f)

        self.epochs = _data.get("epochs", self.epochs)
        self.batch_per_epoch = _data.get("batch_per_epoch", self.batch_per_epoch)
        self.batch_size = _data.get("batch_size", self.batch_size)
        self.optimizer_params = _data.get("optimizer_params", self.optimizer_params.__class__.__name__)
        self.save_model_interval = _data.get("save_model_interval", self.save_model_interval)
        self.device = _data.get("device", self.device)
        self.output_dir = _data.get("output_dir", self.output_dir)
        self.resume_from_checkpoint = _data.get("resume_from_checkpoint", self.resume_from_checkpoint)
        self.accelerator = _data.get("accelerator", self.accelerator)
        self.gradient_scaler = _data.get("gradient_scaler", self.gradient_scaler)
        self.mixed_precision = _data.get("mixed_precision", self.mixed_precision)
        self.training_loss_function = _data.get("training_loss_function", self.training_loss_function)
        self.validation_loss_function = _data.get("validation_loss_function", self.validation_loss_function)

        print(f"Training configuration imported from {import_path}")