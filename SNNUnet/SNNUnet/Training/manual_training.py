"""
Script which implements training with where
user have to adjust the parameters alone. 
Or he needs to provide paths to the config files.
It is "or-or". No mixture.
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
        "--data",
        dest="data_path",
        required=True,
        default = None,
        help="path to the folder containing training and validation data"
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
         path_to_data_fingerprint = _args.fingerprint_path,
         data_path = _args.data_path
         )


# Define main function
def main(export_dir_path = "DefaultDataExportDir",
         device = "cpu",
         path_to_data_fingerprint = None,
         data_path = None):
    """
    Main function which do all the work. Here are predefined config files and it also
    create some output configs. This is how the model is trained if the user wants
    to specify every single parameter of every single component!

    Args:
        * export_dir_path, string,  path to the export dir where all the config 
        files are gonna be created.
        * device, string, device on which methods will we loaded. E.G. cuda:0, 
        cuda:1, cpu...
        * path_to_data_fingerprint, string, path to the dataset fingerprint.
        * data_path, string, path to the directory containing data.
    """
    # Create absolut path to data
    data_path = os.path.abspath(data_path)
    path_to_data_fingerprint = os.path.abspath(path_to_data_fingerprint)

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
    _model_config = ModelConfig(path_to_data_fingerprint = path_to_data_fingerprint,
                                device = device,
                                device_limit = 8000,
                                mixed_precision = True)
    ModelConfig.output_auxiliary_predictions = True
    _model_config.export_to_json(os.path.join(_configs_path,"model_config.json"))
    print(_model_config)
    print("############################################################")

    ###################################################################################
    # Creating loss function
    # Set up loss because we are using custom one.
    print("Building up loss functions!")
    _loss_config = LossConfig_DCandCE()
    LossConfig_DCandCE.loss_name = "DCandCE"
    LossConfig_DCandCE.apply_nonlin = nn.Softmax(dim = 1)
    LossConfig_DCandCE.batch_dice = True
    LossConfig_DCandCE.do_bg = False
    LossConfig_DCandCE.smooth = 1e-5
    LossConfig_DCandCE.clip_tp = None
    LossConfig_DCandCE.weight_dice = 1.0
    LossConfig_DCandCE.weight_ce = 1.0
    LossConfig_DCandCE.ignore_label = -1 #read-out from fingerprint
    _loss_config.export_to_json(os.path.join(_configs_path, "DCandCE_loss_config.json"))
    print(_loss_config)
    # We are using actually deep supervision
    _deep_supervision_loss_config = LossConfigDeepSupervision()
    LossConfigDeepSupervision.name = "dsLoss"
    LossConfigDeepSupervision.loss_function = _loss_config
    LossConfigDeepSupervision.cascade_number = len(_model_config.output_shapes) - 2 # Same as in NNUnet
    _deep_supervision_loss_config.export_to_json(os.path.join(_configs_path, "deep_supervision_loss_config.json"))
    print(_deep_supervision_loss_config)
    print("############################################################")

    ###################################################################################
    # Creating optimizer
    # Optimizer is a default one, all parameters are the same as in the config file
    print("Creating optimizer!")
    _optimizer_config = OptimizerConfig()
    _optimizer_config.export_to_json(os.path.join(_configs_path, "optimizer_config.json"))
    print(_optimizer_config)
    print("############################################################")
    
    ###################################################################################
    # Training configuration
    print("Creating training configuration!")
    _training_config = TrainingConfig()
    TrainingConfig.epochs = 50#1000
    TrainingConfig.batch_per_epoch = 250
    TrainingConfig.batch_size = 2
    TrainingConfig.optimizer_params = _optimizer_config
    TrainingConfig.save_model_interval = 5
    TrainingConfig.training_loss_function = _deep_supervision_loss_config
    TrainingConfig.validation_loss_function = _loss_config
    TrainingConfig.device = device
    TrainingConfig.output_dir = _ckpt_path
    TrainingConfig.accelerator = False
    TrainingConfig.resume_from_checkpoint = None
    TrainingConfig.gradient_scaler = True
    TrainingConfig.mixed_precision = _model_config.mixed_precision
    _training_config.export_to_json(os.path.join(_configs_path, "training_config.json"))
    print(_training_config)
    print("############################################################")
    ###################################################################################
     # Set up Augmentation because we are using custom one, but most of the params are
     # alreadey set just lik we want them
    _augmentation_config = AugmentationConfigNNUnet()
    # Most of the things is already setup so no need to adjust
    AugmentationConfigNNUnet.s_ds_shapes_list = _model_config.output_shapes[:-1]
    AugmentationConfigNNUnet.s_ds_resize = True
    AugmentationConfigNNUnet.c_center_crop = True
    AugmentationConfigNNUnet.c_cc_desired_patch_shape = _model_config.input_shape
    _augmentation_config.export_to_json(os.path.join(_configs_path, "augmentation_config.json"))
    print(_augmentation_config)
    print("############################################################")
    # Build augmentationt method
    _augmentation_method = NNUnetAugmentation(augmentation_config = _augmentation_config)
    ###################################################################################

    ################################################################################
    # Datasets - training dataset
    _dataset_confing = DatasetConfig()
     # Training dataset
    DatasetConfig.ct_path = os.path.join(data_path, "training_data", "img")
    DatasetConfig.seg_path = os.path.join(data_path, "training_data", "seg")
    DatasetConfig.loc_path = os.path.join(data_path, "training_data", "loc")
    DatasetConfig.data_fingerprint_path = path_to_data_fingerprint
    DatasetConfig.patch_shape = _model_config.input_shape
    DatasetConfig.minimum_fg = 1e-7
    DatasetConfig.sample_limit_count = 1500
    DatasetConfig.multiprocessing = 30
    DatasetConfig.augmentation_method = _augmentation_method
    DatasetConfig.keep_in_memory = False
    DatasetConfig.sample_limit = None
    DatasetConfig.force_valid_locations = False
    DatasetConfig.include_background_as_class = False
    DatasetConfig.white_list_cases = None
    print(_dataset_confing)
    _dataset_confing.export_to_json(os.path.join(_configs_path, "training_dataset_config.json"))
    # Create dataset
    _train_dataset = PatchDataset(_dataset_confing)

    # Training Sampler
    _sampler_config = SamplerConfig()
    SamplerConfig.batch_size = _training_config.batch_size
    SamplerConfig.fg_sample_ratio = 0.33
    SamplerConfig.dataset = _train_dataset
    SamplerConfig.class_sampling_dict = None#_path_to_data_fingerprint#None
    SamplerConfig.background_as_random = None#0.33
    print(_sampler_config)
    _sampler_config.export_to_json(os.path.join(_configs_path, "training_sampler_config.json"))
    # Create Sampler
    _sampler = NNUnetSampler(_sampler_config)

    # Training Dataloader 
    _training_dataloader_config = LoaderConfig()
    LoaderConfig.dataset = _train_dataset
    LoaderConfig.sampler = _sampler
    LoaderConfig.number_of_workers = 8
    LoaderConfig.prefetch_factor = 4
    LoaderConfig.use_gpu = True
    LoaderConfig.device = device
    LoaderConfig.collate_fn = custom_collate_fn_for_infinite_dataloader 
    print(_training_dataloader_config)
    _training_dataloader_config.export_to_json(os.path.join(_configs_path, "training_dataloader_config.json"))
    # Dataloader
    _training_dataloader = infinite_dataloader(_training_dataloader_config)
    print("############################################################")
    ################################################################################

    ################################################################################
    # Now validation setup
    """
    # This is validation with moving window!!!
    _dataset_confing = DatasetConfig()
    DatasetConfig.ct_path = os.path.join(data_path, "validation_data", "img")
    DatasetConfig.seg_path = os.path.join(data_path, "validation_data", "seg")
    DatasetConfig.loc_path = os.path.join(data_path, "validation_data", "loc")
    DatasetConfig.patch_shape = _model_config.input_shape
    DatasetConfig.multiprocessing = 30
    DatasetConfig.sample_limit = None
    DatasetConfig.keep_in_memory = 1.0
    DatasetConfig.augmentation_method = None
    DatasetConfig.force_valid_locations = True
    DatasetConfig.include_background_as_class = False
    DatasetConfig.white_list_cases = None
    print(_dataset_confing)
    _dataset_confing.export_to_json(os.path.join(_configs_path, "validation_dataset_config.json"))
    # Create dataset
    _validation_dataset = SingleSampleDataset(_dataset_confing)
    
    # Dataloader for validation
    _validation_dataloader_config = LoaderConfig()
    LoaderConfig.dataset = _validation_dataset
    LoaderConfig.number_of_workers = 8
    LoaderConfig.prefetch_factor = 4
    LoaderConfig.use_gpu = True
    LoaderConfig.device = device
    print(_validation_dataloader_config)
    _validation_dataloader_config.export_to_json(os.path.join(_configs_path, "validation_dataloader_config.json"))
    # Datalaoder
    _validation_dataloader = one_sample_dataloader(_validation_dataloader_config)
    print("############################################################")
    """
    # Datasets - Validation dataset
    _dataset_confing = DatasetConfig()
     # Validation dataset
    DatasetConfig.ct_path = os.path.join(data_path, "validation_data", "img")
    DatasetConfig.seg_path = os.path.join(data_path, "validation_data", "seg")
    DatasetConfig.loc_path = os.path.join(data_path, "validation_data", "loc")
    DatasetConfig.data_fingerprint_path = path_to_data_fingerprint
    DatasetConfig.patch_shape = _model_config.input_shape
    DatasetConfig.minimum_fg = 1e-7
    DatasetConfig.sample_limit_count = 1500
    DatasetConfig.multiprocessing = 30
    DatasetConfig.augmentation_method = None
    DatasetConfig.keep_in_memory = False
    DatasetConfig.sample_limit = None
    DatasetConfig.force_valid_locations = False
    DatasetConfig.include_background_as_class = False
    DatasetConfig.white_list_cases = None
    print(_dataset_confing)
    _dataset_confing.export_to_json(os.path.join(_configs_path, "validation_dataset_config.json"))
    # Create dataset
    _validation_dataset = PatchDataset(_dataset_confing)

    # Validation Sampler
    _sampler_config = SamplerConfig()
    SamplerConfig.batch_size = None
    SamplerConfig.fg_sample_ratio = None
    SamplerConfig.dataset = _validation_dataset
    SamplerConfig.class_sampling_dict = None
    SamplerConfig.background_as_random = None
    SamplerConfig.number_of_samples = 20
    print(_sampler_config)
    _sampler_config.export_to_json(os.path.join(_configs_path, "validation_sampler_config.json"))
    # Create Sampler
    _sampler = ValidationSampler(_sampler_config)

    # Validation Dataloader 
    _validation_dataloader_config = LoaderConfig()
    LoaderConfig.dataset = _validation_dataset
    LoaderConfig.sampler = _sampler
    LoaderConfig.number_of_workers = 1
    LoaderConfig.prefetch_factor = 1
    LoaderConfig.use_gpu = True
    LoaderConfig.device = device
    LoaderConfig.collate_fn = custom_collate_fn_for_sampler_for_validation 
    print(_validation_dataloader_config)
    _validation_dataloader_config.export_to_json(os.path.join(_configs_path, "validation_dataloader_config.json"))
    # Dataloader
    _validation_dataloader = infinite_dataloader(_validation_dataloader_config)
    print("############################################################")
    ################################################################################
    
    #for _item in _validation_dataloader:
    #    print(_item[0].shape, _item[1].shape)
        
    #exit(1)

    # Start training
    _my_training_driver = MainTrainingClass(train_dataloader = _training_dataloader,
                                            valid_dataloader = _validation_dataloader,
                                            model_config = _model_config,
                                            training_config = _training_config)
    _my_training_driver.start_training()


if __name__ == "__main__":
    cli()

    