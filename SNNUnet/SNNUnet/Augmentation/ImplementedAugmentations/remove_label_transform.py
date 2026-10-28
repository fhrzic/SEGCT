""" Implementation of low resolution simulation"""
from typing import Tuple
import torchio as tio
import torch
from torch.nn.functional import interpolate


class RemoveLabel(tio.Transform):
    """
    Class which implements random low resoltuion. How it does it. 
    Select_channel_of_iamge--> chose random scale per axis -->interpolate to that randomly chosen 
    dimension --> interpolate back to original dimension.
    """
    def __init__(self, 
                source_label: int = None,
                destination_label: int = None,
                p: float = 1.0):
        """
        Init function which transforms source label to destination label. It is simple value swap 
        which is usefull to change -1 paddings to 0 in segmentation masks. As you can assume, this
        is applied only to the segmentation mask.

        Args:
            * source_label, int, label which is going to be changed
            * destination_label, int, label to which the source label will be changed
            * p, float, general probability that the whole bluring happens
        """
        # Inherit
        super().__init__(p = p)
        
        # Set labels
        assert source_label is not None or destination_label is not None, \
            "Error, source_label and destination_label can not be none."
        self.source_label = source_label
        self.destination_label = destination_label

    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        """
        Method which applies transform. 

        Args:
            * subject, tio.Subject, subject to which the  transformation will be applied

        Returns:
            * transformed subject
        """
        # Get image as tensor
        if isinstance(subject, tio.Subject):
            _label = subject['label'].data
        else:
            if isinstance(subject, tio.ScalarImage):
                _label = subject.data
            else:
                _label = subject

        # Change _label
        _label = _label.masked_fill_(_label == self.source_label, self.destination_label)

        # Build back subject and return
        if isinstance(subject, tio.Subject):
            subject['label'].data = _label
            return subject
        else:
            if isinstance(subject, tio.ScalarImage):
                subject.data = _label
                return subject
            else:
                return _label
            