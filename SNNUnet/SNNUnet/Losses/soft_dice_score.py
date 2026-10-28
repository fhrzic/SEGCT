from torch import nn
from typing import Callable
import torch

class SoftDiceLoss(nn.Module):
    """
    Class which implements softDiceLoss based on the nnUnet SoftDiceLoss. 
    """
    def __init__(self, 
                 apply_nonlin: Callable = None, 
                 batch_dice: bool = False, 
                 do_bg: bool = True, 
                 smooth: float = 1.,
                 clip_tp: float = None,
                 ignore_label: int = -100):
        """
        Init function for the SoftDiceLoss. It has various flags which default values are set according to nnUnet call.
        Every flag will be explained in further text. I have ommited ignore_label thing because I did not find that 
        necessary in my experiments. But here is the place to add "mask" which actually removes labels that we want
        to neglect during the training.

        Args:
            * apply_nonlin, Callable, if for some reason we want to provide probabilities, we can provide a function here 
            which applies probability such as Sigmoid or Softmax. For instance, for compound loss it is: torch.softmax(x, 1)
            * batch_dice, bool, if set to True the whole batch will be treated as a single image, not as separated cases coming
            from different samples. If False, dice score will be caulcualte per sample. Calculating on the batch level might 
            smooth the variance in small datasets. In the nnUnet (default_experiment_planner.py) this are the values: 3Dlow_res = False, 
            3Dful_res = True (in case of cascade), otherwise it is False. 2D = True. So in our case (fullres3D variant no cascade) this
            param is set to False.
            * do_bg, bool, If set to true, dice loss will include the background into calculation. Otherwise, it will calculate
            loss only based on foreground pixels (casses). Default: False
            * smooth, float, factor to avoid division by 0. Default 1.0 but in nnUnet it was passed as 1e-5
            * clip_tp, float, This can help to prevent minimum TP being 0 in some cases of highly unbalanced data.
              Default: not provided so it is None
            * ignore_label, int, label which is going to be ignore during the calculation of the loss

        DISCLAIMER: I TESTED IT FOR BINARY CLASSES AND IT DOES NOTWORK

        """
        # Inherit
        super(SoftDiceLoss, self).__init__()

        # Set self variables to be used in forward pass
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ignore_label = ignore_label

    def __get_tp_fp_fn_tn(self, 
                          net_output:torch.tensor = None, 
                          gt: torch.tensor = None, 
                          axes: int = None, 
                          square: bool = False,
                          ignore_mask: torch.tensor = None):
        """
        Method which calculates tp, tn and fp for the given arrays
        
        Args:
            * net_output, torch.tensor, prediction of the model. Must be (b, c, x, y(, z)))
            * gt, torch.tensor,  Must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
            * axes, int, in which axes the data is stored. Like last three, or last 2 dimensions
            * square, boolean, indicates if fp, tp and fn will be squared before summation
            * ignore_mask, torch.tensor, this is mask which takes into account ignore_labels. It must have shape ((b, 1, x, y(, z))))

        Returns,
            number of tp, fp, fn, tn in the given net_output and gt.

        # BE CAREFULL IF WE THERE ARE ONLY 2 CLASSES TO SET CORRECT FLAGS. 
        """
        # If axes are not provided, we assume that it is going to be b, c, x , y (z) and we need x, y, z.
        if axes is None:
            axes = tuple(range(2, net_output.ndim))

        # Reshape the volumes properly
        with torch.no_grad():
            # Check if the outputs are compatible, if not, make them compatible
            if net_output.ndim != gt.ndim:
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))
            
            # Make one hot encoding of the output tensor. Namely, the output of nn will be most likely
            # passed through softmax, so one encoding makes perfect sense.
            if net_output.shape == gt.shape:
                # if this is the case then gt is probably already a one hot encoding
                _y_onehot = gt
            else:
                # This means it is not onehot encoded and it should be
                _y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
                _y_onehot.scatter_(1, gt.long(), 1)

        # Calculate tp, fp, fn and tn by multiplying with the one hot encodings
        _tp = net_output * _y_onehot
        _fp = net_output * (~_y_onehot) # this should be 1-_y_onehot but that yeilds wrong result for 2 classes (background + 1 class)
        _fn = (1 - net_output) * _y_onehot
        _tn = (1 - net_output) * (~_y_onehot)

        # Apply mask (One more multiplication...)
        with torch.no_grad():
            _ignore_label_mask = torch.tile(ignore_mask, (1, _tp.shape[1], *[1 for _ in range(2, _tp.ndim)]))
        _tp *= _ignore_label_mask
        _fp *= _ignore_label_mask
        _fn *= _ignore_label_mask
        _tn *= _ignore_label_mask

        # If necessary, it can be squared
        if square:
            _tp = _tp ** 2
            _fp = _fp ** 2
            _fn = _fn ** 2
            _tn = _tn ** 2
        
        # Make final calculation of tp, fp fn and tn
        if len(axes) > 0:
            _tp = _tp.sum(dim=axes, keepdim=False)
            _fp = _fp.sum(dim=axes, keepdim=False)
            _fn = _fn.sum(dim=axes, keepdim=False)
            _tn = _tn.sum(dim=axes, keepdim=False)
        # Return
        return _tp, _fp, _fn, _tn

    def forward(self, x, y):
        # Obtain shape of x
        _shp_x = x.shape

        # If batch_dice is set to true, we merge all data into "one image", so the axes
        # if self.batch_dice is true should involved also the first axis (0) which represents
        # samples. E.G.: 4,7,128,128,128 (4 samples in batch, with 7 clases where is sample is 128x128x128), axes
        # to sum are 0, 2, 3, 4. Otherwise, it is 2, 3 ,4 
        if self.batch_dice:
            _axes = [0] + list(range(2, len(_shp_x)))
        else:
            _axes = list(range(2, len(_shp_x)))

        # Applying nonlinearity here (if provided)
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Calculate ignore mask based on provided ignore_label
        _ignore_mask =  y != self.ignore_label
        # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
        # ignore gradients in those areas anyway
        _masked_y = torch.where(_ignore_mask, y, 0)
        # Calcualte number of fg, this is important for CE loss to avoid caluclating it when there is nothing to calculate
        # e.g. whole patch is egnored
        _num_fg = _ignore_mask.sum() 

        # Add channel 
        _ignore_mask = _ignore_mask.unsqueeze(1)
        # Obtain values necessary to calculate the dice score
        _tp, _fp, _fn, _ = self.__get_tp_fp_fn_tn(net_output = x, 
                                                  gt = _masked_y, 
                                                  axes = _axes, 
                                                  square = False,
                                                  ignore_mask = _ignore_mask)
        # If TP tends to be 0 very often, we can clip it to som small value.
        if self.clip_tp is not None:
            _tp = torch.clip(_tp, min=self.clip_tp , max=None)
        
        # Calculate dice score nominators and denominators
        _nominator = 2 * _tp
        _denominator = 2 * _tp + _fp + _fn

        # Caluclate dices score with aplied smoothing and avoiding division by 0
        _dice_loss = (_nominator + self.smooth) / (torch.clip(_denominator + self.smooth, 1e-8))

        # Remove backgorund from the final loss if necessary
        if not self.do_bg:
            if self.batch_dice:
                _dice_loss = _dice_loss[1:]
            else:
                _dice_loss = _dice_loss[:, 1:]

        # Calculate mean dice loss (acrros all classes)
        _dice_loss = _dice_loss.mean()

        # Return 
        return -_dice_loss, _num_fg