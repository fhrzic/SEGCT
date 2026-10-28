""" 
Script which implements random rotation
TorchIO have random rotation implemented in the random_affine.
2 months I was searching for the error, guess what, it does not work!
Here is my implementation
"""
from typing import Tuple
import torch
import torch.nn.functional as F
import torchio as tio

class Rotation(tio.Transform):
    """
    Class which implements random rotation on both image and segmentation map
    """
    def __init__(
        self,
        rotation_range: Tuple[int] = None,
        p: float = 1):
        """
        Init function for contrast transform.

        Args:
            * rotation range, list of two integers, the random angle will be chosen from the range defined by the 
            two provided integers. The angles are given in degrees; E.G [-30, 30] will chose a random angle between
            -30 and 30
            * p, float, probability to apply this transform, this should be set to 1 because
            we always want to apply it in case that we used oversized input shape
        """

         # Inherit
        super().__init__(p = p)

        # Set desired shape
        self.rotation_range = rotation_range

    def apply_transform(
            self,
            subject: tio.Subject) -> tio.Subject:
        """
        Applies random rotation transformation.
        Args:
            * subject, tio.Subject, subject to which the transformation will be applied.
        Returns:
            * Transformed/rotated subject.
        """
        assert isinstance(subject, tio.Subject), "Input must be a tio.Subject"
        
        # Extract image and label tensors
        _label = subject['label'].data
        _image = subject['image'].data

        # Generate random rotation angles (convert degrees to radians)
        _rotation_angles = torch.FloatTensor(3).uniform_(self.rotation_range[0], self.rotation_range[1]) * torch.pi / 180

        # Rotation matrices for X, Y, Z axes
        _rotation_x = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(_rotation_angles[0]), -torch.sin(_rotation_angles[0])],
                                    [0, torch.sin(_rotation_angles[0]), torch.cos(_rotation_angles[0])]])

        _rotation_y = torch.tensor([[torch.cos(_rotation_angles[1]), 0, torch.sin(_rotation_angles[1])],
                                    [0, 1, 0],
                                    [-torch.sin(_rotation_angles[1]), 0, torch.cos(_rotation_angles[1])]])

        _rotation_z = torch.tensor([[torch.cos(_rotation_angles[2]), -torch.sin(_rotation_angles[2]), 0],
                                    [torch.sin(_rotation_angles[2]), torch.cos(_rotation_angles[2]), 0],
                                    [0, 0, 1]])

        # Compute final rotation matrix (Z * Y * X)
        _rotation = _rotation_z @ _rotation_y @ _rotation_x  # Shape: (3,3)

        # Convert 3x3 rotation matrix to 3x4 by adding a zero translation column
        _rotation = torch.cat([_rotation, torch.zeros(3, 1)], dim=1)  # Shape: (3, 4)

        # Ensure batch size (N=1)
        _rotation = _rotation.unsqueeze(0)  # Shape: (1, 3, 4)

        # Get image shape for grid (add batch & channel dim)
        shape = _image.unsqueeze(0).shape  # Shape: (1, C, D, H, W)

        # Generate affine grid
        _grid = F.affine_grid(_rotation, shape, align_corners=False)

        # Apply rotation using grid_sample
        _rotated_image = F.grid_sample(_image.unsqueeze(0), _grid, mode='bilinear', padding_mode='border', align_corners=False)
        _rotated_label = F.grid_sample(_label.float().unsqueeze(0), _grid, mode='nearest', padding_mode='border', align_corners=False).long()

        # Remove extra dimensions
        subject['label'].data = _rotated_label.squeeze(0)
        subject['image'].data = _rotated_image.squeeze(0)

        # Return
        return subject