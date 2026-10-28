"""
Script which implements training from
existing config files. The idea is:
1.) We have trained model with manual_training.py
script.
2.) There is Configs directory with all configurations.
3.) We made some changes in some of the jsons and want
to train with those configs while keeping everything
else the same. (Manual changes)
"""

# Imports
import sys
import os
import argparse
import shutil
import torch.nn as nn
# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from Configurations.config_model import *
from Configurations.config_loss import *
from Configurations.config_training import *
from Configurations.config_optimizer import *
from Configurations.config_augmentation import *
from Configurations.config_data import *

from Losses.soft_dice_score import *
from Losses.compound_loss_DC_and_CE import *
from Losses.deep_supervision_loss import *
from Augmentation.Utils.nn_unet_augmentation import NNUnetAugmentation
from Data.Dataset.NNUnet_Dataset.dataset import * 
from Data.Sampler.sampler_nnUnet_style import *
from Data.Sampler.sampler_for_validation import *
from Data.Dataloader.dataloader import *
from Data.Dataset.SingleSample_Dataset.dataset import * 
from Training.training_class import MainTrainingClass

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="Process an XLSX file and export results to a specified directory."
    )

    _parser.add_argument(
        "--export",
        dest="export_dir_path",
        required=True,
        help="Path to the export directory (e.g., /Desktop/MyProject/Data)"
    )

    _parser.add_argument(
        "--device",
        dest="device",
        required=True,
        default="cpu",
        help="Name of the device 'cpu' or 'cuda:0', 'cuda:1'..."
    )

    _parser.add_argument(
        "--configurations",
        dest="configurations_path",
        required=True,
        default = None,
        help="path to the folder containing all configurations"
    )
    
    _parser.add_argument(
        "--fingerprint",
        dest="fingerprint_path",
        required=True,
        default = None,
        help="path to the dataset fingerprint"
    )

    # Parse arguments
    _args = _parser.parse_args()
   

    # Call main
    main(export_dir_path=_args.export_dir_path,
         device = _args.device,
         configurations_path = _args.configurations_path,
         path_to_data_fingerprint= _args.fingerprint_path
         )


# Define main function
def main(export_dir_path = "DefaultDataExportDir",
         device = "cpu",
         path_to_data_fingerprint = None,
         configurations_path = None):
    """
   Idea of this main script is that we want to train the model with similar 
   config as we have trained some other model, but change just some stuff. 
   For instance we want to change 

    Args:
        * export_dir_path, string,  path to the export dir where all the config 
        files are gonna be created.
        * device, string, device on which methods will we loaded. E.G. cuda:0, 
        cuda:1, cpu...
        * configurations_path, string, path to the directory containing Configs. This 
        should be folder which contains Configs files produced by manual_training.py because
        naming is IMPORTANT!.
        * path_to_data_fingerprint, string, path to the dataset fingerprint.
    """   
    # Create directory (delete if egsists)
    if os.path.exists(export_dir_path):
        print(f"Directory exists. Removing: {export_dir_path}")
        shutil.rmtree(export_dir_path)

    os.makedirs(export_dir_path)

    # Create config file inside of it
    _configs_path = os.path.join(export_dir_path, "Configs")
    os.makedirs(_configs_path)

    # Create Checkpoints dir
    _ckpt_path = os.path.join(export_dir_path, "Checkpoints")
    os.makedirs(_ckpt_path)

    print(f"Export Directory Creation: {export_dir_path}")
    print(f"Export Directory Creation: {_configs_path}")
    print(f"Export Directory Creation: {_ckpt_path}")
    print("############################################################")

    #################################################################################
    # First we build model based on the provided fingerprint
    print("Building model based on the provided fingerprint!")
    # Model Config
    _model_config = ModelConfig()
    _model_config.import_from_json(os.path.join(configurations_path,"model_config.json"))
    print(_model_config)
    _model_config.export_to_json(os.path.join(_configs_path, "model_config.json"))

    print("############################################################")

    ###################################################################################
    # Creating loss function
    # Set up loss because we are using custom one.
    print("Building up loss functions!")
    _loss_config = LossConfig_DCandCE()
    _loss_config.import_from_json(os.path.join(configurations_path, "DCandCE_loss_config.json"))
    _loss_config.apply_nonlin = nn.Softmax(dim = 1)
    print(_loss_config)
    _loss_config.export_to_json(os.path.join(_configs_path, "DCandCE_loss_config.json"))

    if os.path.exists(os.path.join(configurations_path, "deep_supervision_loss_config.json")):
        # We are using actually deep supervision
        _deep_supervision_loss_config = LossConfigDeepSupervision()
        _deep_supervision_loss_config.import_from_json(os.path.join(configurations_path, "deep_supervision_loss_config.json"))
        _deep_supervision_loss_config.loss_function = _loss_config
        _deep_supervision_loss_config.export_to_json(os.path.join(_configs_path, "deep_supervision_loss_config.json"))
        print("############################################################")

    ###################################################################################
    # Creating optimizer
    # Optimizer is a default one, all parameters are the same as in the config file
    print("Creating optimizer!")
    _optimizer_config = OptimizerConfig()
    _optimizer_config.import_from_json(os.path.join(configurations_path, "optimizer_config.json"))
    print(_optimizer_config)
    _optimizer_config.export_to_json(os.path.join(_configs_path, "optimizer_config.json"))
    print("############################################################")
    
    ###################################################################################
    # Training configuration
    print("Creating training configuration!")
    _training_config = TrainingConfig()
    _training_config.import_from_json(os.path.join(configurations_path, "training_config.json"))
    _training_config.optimizer_params = _optimizer_config
    _training_config.training_loss_function = _deep_supervision_loss_config 
    _training_config.validation_loss_function = _loss_config
    _training_config.device = device
    print(_training_config)
    _training_config.export_to_json(os.path.join(_configs_path, "training_config.json"))
    print("############################################################")

    ###################################################################################
     # Set up Augmentation because we are using custom one, but most of the params are
     # alreadey set just lik we want them
    _augmentation_config = AugmentationConfigNNUnet()
    # Most of the things is already setup so no need to adjust
    _augmentation_config.import_from_json(os.path.join(configurations_path, "augmentation_config.json"))
    _augmentation_config.export_to_json(os.path.join(_configs_path, "augmentation_config.json"))

    print("############################################################")
    # Build augmentationt method
    _augmentation_method = NNUnetAugmentation(augmentation_config = _augmentation_config) if _augmentation_config.name == "NNUnet" else None
    ###################################################################################
    print(_augmentation_method)
    ################################################################################
    # Datasets - training dataset
    _dataset_config = DatasetConfig()
    # Training dataset
    _dataset_config.import_from_json(os.path.join(configurations_path, "training_dataset_config.json"))
    _dataset_config.patch_shape = _model_config.input_shape
    _dataset_config.augmentation_method = _augmentation_method
    print(_dataset_config)
    _dataset_config.export_to_json(os.path.join(_configs_path, "training_dataset_config.json"))
    # Create dataset
    print(_dataset_config)
    _train_dataset = PatchDataset(_dataset_config) 

    # Training Sampler
    _sampler_config = SamplerConfig()
    _sampler_config.import_from_json(os.path.join(configurations_path, "training_sampler_config.json"))
    _sampler_config.batch_size = _training_config.batch_size
    _sampler_config.dataset = _train_dataset
    print(_sampler_config)
    _sampler_config.export_to_json(os.path.join(_configs_path, "training_sampler_config.json"))
    # Create Sampler
    _sampler = NNUnetSampler(_sampler_config)

    # Training Dataloader 
    _training_dataloader_config = LoaderConfig()
    _training_dataloader_config.import_from_json(os.path.join(configurations_path, "training_dataloader_config.json"))
    _training_dataloader_config.dataset = _train_dataset
    _training_dataloader_config.sampler = _sampler
    _training_dataloader_config.device = device
    LoaderConfig.collate_fn = custom_collate_fn_for_infinite_dataloader 
    print(_training_dataloader_config)
    _training_dataloader_config.export_to_json(os.path.join(_configs_path, "training_dataloader_config.json"))
    # Dataloader
    _training_dataloader = infinite_dataloader(_training_dataloader_config)
    print("############################################################")
    ################################################################################

    # Datasets - Validation dataset
    _dataset_confing = DatasetConfig()
    _dataset_confing.import_from_json(os.path.join(configurations_path, "validation_dataset_config.json"))
    DatasetConfig.patch_shape = _model_config.input_shape
    DatasetConfig.augmentation_method = NNUnetAugmentation(augmentation_config = _augmentation_config) if _dataset_config.augmentation_method == "NNUnet" else None
    print(_dataset_confing)
    _dataset_config.export_to_json(os.path.join(_configs_path, "validation_dataset_config.json"))
    
    # Create dataset
    _validation_dataset = PatchDataset(_dataset_confing)

    # Validation Sampler
    _sampler_config = SamplerConfig()
    _sampler_config.import_from_json(os.path.join(configurations_path, "validation_sampler_config.json"))
    SamplerConfig.dataset = _validation_dataset
    print(_sampler_config)
    _sampler_config.export_to_json(os.path.join(_configs_path, "validation_sampler_config.json"))

    # Create Sampler
    _sampler = ValidationSampler(_sampler_config)
    # Validation Dataloader 
    _validation_dataloader_config = LoaderConfig()
    _validation_dataloader_config.import_from_json(os.path.join(configurations_path, "validation_dataloader_config.json"))

    LoaderConfig.dataset = _validation_dataset
    LoaderConfig.sampler = _sampler
    LoaderConfig.device = device
    LoaderConfig.collate_fn = custom_collate_fn_for_sampler_for_validation 
    print(_validation_dataloader_config)
    # Dataloader
    _validation_dataloader = infinite_dataloader(_validation_dataloader_config)
    print("############################################################")
    ################################################################################
    
    # Start training
    _my_training_driver = MainTrainingClass(train_dataloader = _training_dataloader,
                                            valid_dataloader = _validation_dataloader,
                                            model_config = _model_config,
                                            training_config = _training_config)
    _my_training_driver.start_training()


if __name__ == "__main__":
    cli()

    