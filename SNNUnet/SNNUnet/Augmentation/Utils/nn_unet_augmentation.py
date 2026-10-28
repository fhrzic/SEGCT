""" AugmentationMethods same as NN unet """
# Imports
import sys
import os
from typing import List
import torch
import numpy as np

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)
###############################################################

from Configurations.config_augmentation import AugmentationConfigNNUnet
import torchio as tio
from ImplementedAugmentations.gaussian_blur import *
from ImplementedAugmentations.brightness_transform import *
from ImplementedAugmentations.contrast_transform import *
from ImplementedAugmentations.low_resolution import *
from ImplementedAugmentations.gamma_transform import *
from ImplementedAugmentations.deep_supervision_resize import *
from ImplementedAugmentations.center_crop import *
from ImplementedAugmentations.rotation import *
from ImplementedAugmentations.scaling import *
from ImplementedAugmentations.remove_label_transform import *

class NNUnetAugmentation():
    """
    Class for creating augmentation similar to the one in nnUNET
    """
    def __init__(self, augmentation_config: AugmentationConfigNNUnet = None):
        """
        Init function which creates augmentation on the 3D patch. It is insipred 
        and based on the torch nnUnet

        Args:
            * augmentation_config, augmentationConfig, configuration data for 
            realisation of different augmentation metrices.
        """

        # Main list for transformations
        _main_transformation_list = []
        _image_only_transformation_list = []

        # Affine transformations (applies both to image and segmentation mask)
        ## Apply rotation
        if augmentation_config.a_rotation == True:
            _rotation_transform = Rotation(
                rotation_range = augmentation_config.a_r_degrees,
                p = augmentation_config.a_r_probability
            )
            # Add it to main tranformation list
            _main_transformation_list.append(_rotation_transform)

        ## Apply scaling
        if augmentation_config.a_scaling == True:
            _scaling_transform = Scaling(
                scale_range = augmentation_config.a_s_scale,
                apply_same_ratio = augmentation_config.a_s_same_ratio,
                p = augmentation_config.a_s_probability
            )
            # Add it to main tranformation list
            _main_transformation_list.append(_scaling_transform)

        ############################################################################
        # Image only transfrom (no segmentations)
        # Noise transformations
        ## Apply Gausian noise
        if augmentation_config.g_noise == True:
            _gaussian_noise_transform = tio.RandomNoise(
                mean = augmentation_config.g_n_mean,
                std = augmentation_config.g_n_std,
                p = augmentation_config.g_n_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_gaussian_noise_transform)

        ## Apply Gausian blur 
        if augmentation_config.g_blur == True:
            _gaussian_blur_transform = GaussianBlur(
                sigma_range = augmentation_config.g_b_sigma_range,
                probability_per_channel = augmentation_config.g_b_probability_per_channel,
                synchronize_channels = augmentation_config.g_b_synchronize_channels,
                synchronize_axes = augmentation_config.g_b_synchronize_axes,
                p = augmentation_config.g_b_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_gaussian_blur_transform)

        # Intensity transformations
        ## Apply Brightness Transfrom
        if augmentation_config.i_brightness == True:
            _brightness_transform = BrightnessTransform(
                intensity_range = augmentation_config.i_b_intensity_range,
                synchronize_channels = augmentation_config.i_b_synchronize_channels,
                probability_per_channels = augmentation_config.i_b_probability_per_channels,
                p = augmentation_config.i_b_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_brightness_transform)

        ## Apply Contrast Transform
        if augmentation_config.i_contrast == True:
            _contrast_transform = ContrastTransform(
                contrast_range = augmentation_config.i_c_contrast_range,
                preserve_range = augmentation_config.i_c_preserve_range,
                synchronize_channels = augmentation_config.i_c_synchronize_channels,
                probability_per_channels = augmentation_config.i_c_probability_per_channels,
                p = augmentation_config.i_c_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_contrast_transform)

        # Apply Low Resolution transform
        if augmentation_config.r_low_resolution == True:
            _low_resolution_transform = LowResolution(
                scale_range = augmentation_config.r_lr_scale_range,
                synchronize_channels = augmentation_config.r_lr_synchronize_channels,
                synchronize_axes = augmentation_config.r_lr_synchronize_axes,
                probability_per_channels = augmentation_config.r_lr_probability_per_channels,
                p = augmentation_config.r_lr_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_low_resolution_transform)

        # Apply Gamma Transform
        if augmentation_config.i_gamma_1:
            _gamma_transform_1 = GammaTransform(
                gamma_scale = augmentation_config.i_g_1_gamma_scale,
                invert_image = augmentation_config.i_g_1_invert_image,
                synchronize_channels = augmentation_config.i_g_1_synchronize_channels,
                probability_per_channels = augmentation_config.i_g_1_probability_per_channels,
                retain_stats = augmentation_config.i_g_1_retain_stats,
                p = augmentation_config.i_g_1_probability)
            # Add it to image only list
            _image_only_transformation_list.append(_gamma_transform_1)

        # Apply Gamma Transform
        if augmentation_config.i_gamma_2:
            _gamma_transform_2 = GammaTransform(
                gamma_scale = augmentation_config.i_g_2_gamma_scale,
                invert_image = augmentation_config.i_g_2_invert_image,
                synchronize_channels = augmentation_config.i_g_2_synchronize_channels,
                probability_per_channels = augmentation_config.i_g_2_probability_per_channels,
                retain_stats = augmentation_config.i_g_2_retain_stats,
                p = augmentation_config.i_g_2_probability)

            # Add it to image only list
            _image_only_transformation_list.append(_gamma_transform_2)

        # Create composition which applies only to images
        _image_only_transforms = tio.Compose(_image_only_transformation_list)
        _image_only_transforms_l = tio.Lambda(_image_only_transforms, types_to_apply = [tio.IMAGE])
        _main_transformation_list.append(_image_only_transforms_l)

        ############################################################################
        ## Apply Mirror Transform
        if augmentation_config.a_mirror:
            _mirror_transform = tio.RandomFlip(
                axes = augmentation_config.a_m_axes,
                flip_probability = augmentation_config.a_m_probabilty_per_axis,
                p = augmentation_config.a_m_probability)
            # Add it to main tranformation list
            _main_transformation_list.append(_mirror_transform)

        ############################################################################
        ## Swap label in segmentation mask if necessary
        if augmentation_config.r_remove_label_transform:
            _remove_label_transform = RemoveLabel(
                source_label = augmentation_config.r_rlt_source_label,
                destination_label = augmentation_config.r_rlt_destination_label,
                p = augmentation_config.r_rlt_probability)
            # Add it to main tranformation list
            _main_transformation_list.append(_remove_label_transform)


        ############################################################################
        # if desired patch size is not None, and we applied rotation and/or scaling
        # we have to calculate bigger patch size to ensure preservation of the content
        # and do the center crop to the desired patch size
        if augmentation_config.c_center_crop:
            # Calculate initi patch size
            self.initial_patch_size = self.__calculate_initial_patch_size(
                desired_patch_size = augmentation_config.c_cc_desired_patch_shape,
                rotation_angels = augmentation_config.a_r_degrees,
                scale_range = augmentation_config.a_s_scale)
            
            # Since the augmentation got this initial_patch_size as input we need to
            # center crop it to the desired_patch_size as the output
            _center_crop = CenterCrop(
                desired_shape = augmentation_config.c_cc_desired_patch_shape,
                p = augmentation_config.c_cc_probability)
            # Add it to the main transformation list
            _main_transformation_list.append(_center_crop)
        else:
            # This is just to avoid potential errors if someone envokes get_initial_patch_size
            self.initial_patch_size = None

        ############################################################################
        # Last but not least, if deep supervision, we need to retrive different scales
        # of segmentation maks
        if augmentation_config.s_ds_resize:
            _ds_resize_transform = DeepSupervisionResize(
                shapes_list = augmentation_config.s_ds_shapes_list,
                p = augmentation_config.s_ds_probability
            )
            # Add it to main transformation list
            _main_transformation_list.append(_ds_resize_transform)
        ############################################################################

        # Compose full transformation
        self.main_transformation = tio.Compose(_main_transformation_list)

    def __call__(
        self,
        input_patch: torch.Tensor = None,
        segmentation_mask: torch.Tensor = None)->List[torch.Tensor]:
        """
        On call this method creates proper subject which is then passed through 
        defined augmentation methods.

        Args:
            * input_patch, torch.Tensor, patch of shape (1, x, y, z) on which the given transform 
            will be applied
            * segmentation_mask, torch.Tensor, corresponding segmentation mask to the given patch

        Returns:
            * List [torch.Tensor, torch.Tensor], list of two tensors where first represent 
            transformed image, while second represents segmentation mask
        """
        # Create a subject
        _subject = tio.Subject(
            image = tio.ScalarImage(tensor = input_patch),
            label = tio.LabelMap(tensor = segmentation_mask),
            ds_labels = []
        )

        # Apply transform
        _subject = self.main_transformation(_subject)

        # Break it to list
        _subject_as_list = [_subject["image"].data, _subject["label"].data, _subject["ds_labels"]]

        # Return
        return _subject_as_list
    
    def __calculate_initial_patch_size(
        self,
        desired_patch_size: List[int] = None,
        rotation_angels: List[int] = None,
        scale_range: List[int] = None)->List[int]:
        """
        Method which calculates initial patch size needed to obtain trageted_network_patch_size 
        (patch size) with valid region. By valid regions it is ment that after rotation of and scale,
        the initial region is still preserved.

        Args:
            * desired_patch_size, list of integers, three integers representing width, heigh,
            and depth of the patch that will be placed as the input to the neural network.
            * rotation_angels, list of two integers, represents max and mix angles choosen for rotation
            * scale_range, list of two integers, represents max and mins of the scales interval
        
        Returns:
            List of three integers representing patch size which needs to be sampled in order to
            guarrantee the content preservation of the desired_patch_size.
        """
        # Calculate maximum rotation:
        _maximum_rotation = max(np.abs(rotation_angels))
        
        # Safty for too big angels 135 is having random
        _rotation = min(90 , _maximum_rotation)
        
        _x_desired, _y_desired, _z_desired = desired_patch_size
        
        # Rotated
        _rotation_rad = np.radians(_rotation)
        _x_rotated = _x_desired * np.cos(_rotation_rad) + np.sqrt(_y_desired**2 + _z_desired**2) * np.sin(_rotation_rad)
        _y_rotated = _y_desired * np.cos(_rotation_rad) + np.sqrt(_x_desired**2 + _z_desired**2) * np.sin(_rotation_rad)
        _z_rotated = _z_desired * np.cos(_rotation_rad) + np.sqrt(_x_desired**2 + _y_desired**2) * np.sin(_rotation_rad)
        _rotated_shape = np.array([_x_rotated, _y_rotated, _z_rotated])
        
        # Final shape
        _final_shape = _rotated_shape / min(scale_range)
        
        # Return
        return _final_shape.astype(int).tolist()
        
    def get_initial_patch_size(self)->List[int]:
        """
        Public method which returs intial patch size required to ensure that 
        desired patch size always focues on the content.

        Returns:
            * required initial patch size to guarantee the desired patch content
        """
        # Return
        return self.initial_patch_size



