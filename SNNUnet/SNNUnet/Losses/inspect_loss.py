""" Testing script for the losses"""
# Imports
import sys
import os
import argparse
import torch.nn as nn

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from Configurations.config_loss import *
from Losses.soft_dice_score import *
from Losses.compound_loss_DC_and_CE import *
from Losses.deep_supervision_loss import *

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="Simulates loss creation"
    )

    _parser.add_argument(
        "--export_path",
        dest="export_dir_path",
        required=True,
        help="Path to the export directory (e.g., /Desktop/MyProject/Data)"
    )

    _args = _parser.parse_args()

    # Call main
    main(export_dir = _args.export_dir_path)

def main(export_dir: str = None):
    """
    Main funtion which build model from the dataset fingerprint and export all
    necessary files to the export dir including model description.

    Args:
        * export_dir, string, string representing path to the directory where loss configs will be saved 

    Output:
        * print loss configs
    """
    # Create export dir if it does not exists
    os.makedirs(export_dir, exist_ok=True)
    
    # Create loss
        # Config loss
    # Set up loss because we are using custom one.
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
    
    print("######################################\n")
    print(_loss_config)
    # We are using actually deep supervision
    _deep_supervision_loss_config = LossConfigDeepSupervision()
    LossConfigDeepSupervision.name = "dsLoss"
    LossConfigDeepSupervision.loss_function = _loss_config
    LossConfigDeepSupervision.cascade_number = 5 # Same as in NNUnet
    print(_deep_supervision_loss_config)

    print("######################################\n")
    print("Now exporting the losses")
    _loss_config.export_to_json(os.path.join(export_dir, "DCandCE_loss_config.json"))
    _deep_supervision_loss_config.export_to_json(os.path.join(export_dir, "deep_supervision_loss_config.json"))
    print("Exporting done! Check DCandCE_loss_config.json and deep_supervision_loss_config.json in export dir!")
    
    print("######################################\n")
    print("Load from jsons")
    _loss_config = LossConfig_DCandCE()
    _loss_config.import_from_json(os.path.join(export_dir, "DCandCE_loss_config.json"))
    LossConfig_DCandCE.apply_nonlin = nn.Softmax(dim = 1) if _loss_config.apply_nonlin == "Softmax" else None
    
    _deep_supervision_loss_config = LossConfigDeepSupervision()
    _deep_supervision_loss_config.import_from_json(os.path.join(export_dir, "deep_supervision_loss_config.json"))
    LossConfigDeepSupervision.loss_function = _loss_config if LossConfigDeepSupervision.loss_function == "DcandCE" else None
    print("Loaded configs")
    print(_loss_config)
    print(_deep_supervision_loss_config)
    print("######################################\n")
                      
if __name__ == "__main__":
    #_data_fingerprint_path = "/mnt/SSD/Franko/Desktop/Research/DataFingerprint/Scripts/DatasetFingerprintTrainingSetNEW.json"
    #main(data_fingerprint_path = _data_fingerprint_path)
    cli()
