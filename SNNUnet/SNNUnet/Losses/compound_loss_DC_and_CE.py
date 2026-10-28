""" Script which implement all losses """
import os
import sys
from torch import nn
from torch.nn import CrossEntropyLoss
import torch

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from Configurations.config_loss import *
from Losses.soft_dice_score import *

class DC_and_CE_loss(nn.Module):
    """
    Loss wich combines DiceScore and CrossEntropy. It is Loss = weight_dice*DC + weight_ce*CE (weights are provided in config)
    """
    def __init__(self, 
                 config: LossConfig_DCandCE = None):
        """
        Init function of the loss which configures everything about the two loss componentes.

        Args:
            * config, lossConfig_DCandCE, simple class which holds all necessary info to build this loss.
            It is going to be a little "digging", but please see config_loss.py and SoftDiceScore.py scripts
            for more info. There is a lot of parameters, but most of things is straight forward.

        DISCLAIMER: I TESTED IT FOR BINARY CLASSES AND IT DOES NOTWORK (DICE SCORE)
        """
        # Inherit 
        super(DC_and_CE_loss, self).__init__()
        
        # Set weights
        self.weight_dice = config.weight_dice
        self.weight_ce = config.weight_ce

        # Obtain cross entropy. It is simple CE loss from Torch

        self.ce = CrossEntropyLoss(ignore_index = config.ignore_label)

        # This is the SoftDiceLoss implemented in SoftDiceLoss.py
        
        self.dc = SoftDiceLoss(apply_nonlin = config.apply_nonlin, 
                               batch_dice = config.batch_dice, 
                               do_bg = config.do_bg, 
                               smooth = config.smooth,
                               clip_tp = config.clip_tp,
                               ignore_label = config.ignore_label)

    def forward(self, 
        net_output: torch.Tensor, 
        target: torch.Tensor)->float:
        """
        Forward the loss. 

        Args:
            * net_output, torch.Tensor, shape of the tensor b, c, x, y (,z) where b is batch, c is number of classes, x,y,z 
            is patch size.
            * target, torch.Tensor, shape of the tensor is b, c, x, y, (,z) where b is batch , c is 1, and x,y,z is path
            size. Elements in this case are class indices 0, 1, 2, 3...

        Returns:
            * loss for given batch
        """
        # Calculate dc_loss
        _dc_loss, _num_fg = self.dc(net_output, target) if self.weight_dice != 0 else 0
        
        # Calculate ce loss
        _ce_loss = self.ce(net_output, target) if self.weight_ce != 0 and _num_fg > 0 else 0

        # Losses combined
        _loss = self.weight_ce * _ce_loss + self.weight_dice * _dc_loss
        return _loss