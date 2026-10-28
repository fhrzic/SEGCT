""" 
Script which implements center crop
TorchIO does not have center crop so this
is my implementation
"""
from typing import Tuple, List
import torchio as tio
import torch

class CenterCrop(tio.Transform):
    """
    Class which implements center crop
    """
    def __init__(
        self,
        desired_shape: List[int] = None,
        p: float = 1):
        """
        Init function for contrast transform.

        Args:
            * desired_shape, List[int], this is the desired shape which will be cropped
            arround the center voxel.
            * p, float, probability to apply this transform, this should be set to 1 because
            we always want to apply it in case that we used oversized input shape
        """

         # Inherit
        super().__init__(p = p)

        # Set desired shape
        self.desired_shape = desired_shape

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Method which applies transfrom and caluclates all necessery stuff
        Args:
            * subject, tio.Subject, subject to which the  transformation will be applied

        Returns:
            * transformed subject
        """
        # Obtain tensors for label and image. Center crop is applied to both
        assert isinstance(subject, tio.Subject), "This is applied to both label and image, but something is wrong!"        
        _label = subject['label'].data
        _image = subject['image'].data

        # Obtain cropping locations
        # Size of the patch
        _, _x, _y, _z = _image.shape  # (C, x, y, z)
        
        # Desired shape per channel
        _x_d, _y_d, _z_d = self.desired_shape
        
        # Calculate center volume area
        _x1, _y1, _z1 = (_x - _x_d) // 2, (_y - _y_d) // 2, (_z - _z_d) // 2
        _x2, _y2, _z2 = _x1 + _x_d, _y1 + _y_d, _z1 + _z_d
        
        # Apply cropping
        _image = _image[:, _x1:_x2, _y1:_y2, _z1:_z2]
        _label = _label[:, _x1:_x2, _y1:_y2, _z1:_z2]

        # Build back subject and return
        subject['label'].data = _label
        subject['image'].data = _image
        return subject