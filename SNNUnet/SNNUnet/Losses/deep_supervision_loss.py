""" 
Deep supervision loss which is acutally a wrapper around
compound loss. There is special config for this loss.

"""
import sys
import os
from typing import List
import numpy as np
import torch

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from Configurations.config_loss import *
from Losses.soft_dice_score import *
from Losses.compound_loss_DC_and_CE import *

class DeepSupervisionLoss(torch.nn.Module):
    """
    Class which is actually a wrapper around another loss function 
    provided. So it takes one loss function/class + one config which
    specifies the necessary parameters of DeepSupervisionLoss
    """
    def __init__(
        self,
        ds_config: LossConfigDeepSupervision = None):
        """
        Init function which get as input ds_config which is a config file
        from which this loss is builded.

        Args:
            * ds_config, DeepSupervisionConfig, this is a config which contains 
            several crucial informations necessary for building this loss such as
            another loss which is applied per pair of segmentation mask and 
            number of scales to which this loss applies. For
            additional information visit config_loss.py
        """
        # Unpack all necessary variables
        # First one is to build loss -> similar to obtain loss in Training.py
        # Inherit
        super(DeepSupervisionLoss, self).__init__()


        # We need to determine which loss are we applying 
        if isinstance(ds_config.loss_function, str):
            # It is a simple loss
            assert ds_config.loss_function in ["nll", "cross_entropy"], \
                f"Unkown loss function {ds_config.loss_function}. Please check the loss function"
            # Set nll loss
            if ds_config.loss_function == "nll":
                _loss = torch.nn.NLLLoss()
            # Cross entropy
            if ds_config.loss_function == "cross_entropy":
                _loss = torch.nn.CrossEntropyLoss()
        else:
            # Figure out which loss - It is complex one
            assert ds_config.loss_function.loss_name in ["DCandCE"], \
                f"Unkown loss function {ds_config.loss_function.loss_name}. Please check the loss function"
            # DCandCE
            if ds_config.loss_function.loss_name == "DCandCE":
                _loss = DC_and_CE_loss(config = ds_config.loss_function)
        # Assign loss
        self.loss = _loss

        # Determine weights. Number of cascades
        self.weights = np.array([1 / (2 ** _i) 
                                 for _i in range(ds_config.cascade_number)])
        # Normalize weights
        self.weights = self.weights / self.weights.sum()

        # Number of losses
        self.cascade_number = ds_config.cascade_number


    def forward(
        self,
        net_outputs: List[torch.tensor] = None,
        targets: List[torch.tensor] = None)->float:
        """
        Forward the loss. 

        Args:
            * net_output, list of torch tensors, shape of the tensors b, c, x, y (,z) 
            where b is batch, c is number of classes, x,y,z is patch size. 
            * target, list of torch tensors, shape of the tensor is b, c, x, y, (,z) 
            where b is batch , c is 1, and x,y,z is path size.
            Elements in this case are class indices 0, 1, 2, 3...

        Returns:
            * loss for given batch
        """
        net_outputs.reverse()
        targets.reverse()
    
        # Calculate loss
        _loss = sum([self.weights[_i] * self.loss(_prediction, _target)
                     for _i, (_prediction, _target) in enumerate(zip(
                        net_outputs[0:self.cascade_number],
                        targets[0:self.cascade_number]))])
        return _loss
