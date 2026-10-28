""" 
Script which implements random scaling.
TorchIO does not have a dedicated random scaling transform,
so here is my implementation.
"""
from typing import Tuple
import torch
import torch.nn.functional as F
import torchio as tio

class Scaling(tio.Transform):
    """
    Class which implements random scaling on both image and segmentation map.
    """
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        apply_same_ratio: bool = True,
        p: float = 1):
        """
        Init function for scaling transform.

        Args:
            * scale_range, Tuple of two floats defining the minimum and maximum scale factors.
            * apply_same_ratio, boolean, if set to True, same scale will be use along all axes
            * p: float, probability to apply this transform.
        """
        super().__init__(p=p)

        # Set scale range
        self.scale_range = scale_range

        # Set ratio
        self.apply_same_ratio = apply_same_ratio

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Applies random scaling transformation.
        Args:
            * subject: tio.Subject, subject to which the transformation will be applied.
        Returns:
            * Transformed/scaled subject.
        """
        assert isinstance(subject, tio.Subject), "Input must be a tio.Subject"
        
        # Extract image and label tensors
        _label = subject['label'].data
        _image = subject['image'].data

        # Generate random scale factors for each axis, or same for all axis
        if self.apply_same_ratio:
            scale_factors = torch.FloatTensor(1).uniform_(self.scale_range[0], self.scale_range[1])
            scale_factors = scale_factors.expand(3)
        else:
            scale_factors = torch.FloatTensor(3).uniform_(self.scale_range[0], self.scale_range[1])
        
        # Create scaling matrix (3x3)
        _scaling = torch.diag(scale_factors)
        
        # Convert 3x3 scaling matrix to 3x4 by adding a zero translation column
        _scaling = torch.cat([_scaling, torch.zeros(3, 1)], dim=1)  # Shape: (3, 4)
        
        # Ensure batch size (N=1)
        _scaling = _scaling.unsqueeze(0)  # Shape: (1, 3, 4)
        
        # Get image shape for grid (add batch & channel dim)
        shape = _image.unsqueeze(0).shape  # Shape: (1, C, D, H, W)
        
        # Generate affine grid
        _grid = F.affine_grid(_scaling, shape, align_corners=False)
        
        # Apply scaling using grid_sample
        _scaled_image = F.grid_sample(_image.unsqueeze(0), _grid, mode='bilinear', padding_mode='border', align_corners=False)
        _scaled_label = F.grid_sample(_label.float().unsqueeze(0), _grid, mode='nearest', padding_mode='border', align_corners=False).long()
        
        # Remove extra dimensions
        subject['label'].data = _scaled_label.squeeze(0)
        subject['image'].data = _scaled_image.squeeze(0)
        
        # Return
        return subject