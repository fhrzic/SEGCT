"""
Config for the augmentation of the NNUnet 
"""
import json
import os

class AugmentationConfigNNUnet:
    '''
    Class for specifying augmentation on the training set data. It is strongly based on the nnUNET
    No json export or import for this.It is to be implemented.
    '''

    '''
    aug_name: augumentation name, usefull to difirentiet between all the possible augmentations. This 
    is a must.
    '''
    name = "NNUnet"

    ##############################################
    # Random affine, rotation, scaling, and mirror
    a_rotation = True # Indicator if rotation will be applied
    a_r_degrees = (-30, 30) # (a,b), uniformly chosen angle from a to b
    a_r_probability = 0.2 # Probability to apply

    a_scaling = True # Indicator if scaling should be applied
    a_s_scale = (0.7, 1.4) # Scalling
    a_s_same_ratio = True # Applay one ratio to all axes
    a_s_probability = 0.2 # Probabiliy to apply

    a_mirror = True # Indicator if scaling should be applied
    a_m_axes = (0,1,2) # Axes on which the random flip will be applied
    a_m_probabilty_per_axis = 0.5 # Probability that the axis will be flipped
    a_m_probability = 1 # Probability to apply

    ##############################################
    # Random Gaussians: Noise and Blur
    g_noise = True # Indicator if noise will be applied
    g_n_mean = 0 # Mean is zero
    g_n_std = (0, 0.1) # Range for standard deviations
    g_n_probability = 0.1 # Probability to apply

    g_blur = True # Indicator if blur will be applied
    g_b_sigma_range = (0.5, 1.) # Standard deviaton of random blur
    g_b_probability_per_channel = 0.5 # Chances that blur is appled for chanel
    g_b_synchronize_channels = False # Same sigma accross channels
    g_b_synchronize_axes = False # Same sigma accross axes
    g_b_probability = 0.2 # Probability to apply

    ##############################################
    # Random Intensities: Brightness, Contrast and Gamma
    i_brightness = True # Indicator if brightness should be applied
    i_b_intensity_range = (0.75, 1.25) # Brigthenss multiplier range
    i_b_synchronize_channels = False # Apply same multiplier accross all ranges
    i_b_probability_per_channels = 1 # Proability to apply on each channel
    i_b_probability = 0.15 # Probability to apply

    i_contrast = True # Indicator if contrast should be applied
    i_c_contrast_range = (0.75, 1.25) # Contrast range (it has specific policy on how it is applies. Please check)
    i_c_preserve_range = True # Min max value will be clamped if set to true
    i_c_synchronize_channels = False # Apply same contrast accross all chanels
    i_c_probability_per_channels = 1 # Proability to apply on each channel
    i_c_probability = 0.15 # Probability to apply

    i_gamma_1 = True # Indicator if gamma should be applied
    i_g_1_gamma_scale = (0.7, 1.5) # gamma scale as a multiplier (it has specific policy that applies, please check it out)
    i_g_1_invert_image = True # Inverting the image (the invertion is applied backwards also)
    i_g_1_synchronize_channels = False # Apply same sampled gamma value accross all channels
    i_g_1_probability_per_channels = 1 # Probiabiliy to apply on each channel
    i_g_1_retain_stats = True # Retain same mean and std after applyin gamma correction
    i_g_1_probability = 0.1 # Probability to apply

    i_gamma_2 = True # Indicator if gamma should be applied
    i_g_2_gamma_scale = (0.7, 1.5) # gamma scale as a multiplier (it has specific policy that applies, please check it out)
    i_g_2_invert_image = False # Inverting the image (the invertion is applied backwards also)
    i_g_2_synchronize_channels = False # Apply same sampled gamma value accross all channels
    i_g_2_probability_per_channels = 1 # Probiabiliy to apply on each channel
    i_g_2_retain_stats = True # Retain same mean and std after applyin gamma correction
    i_g_2_probability = 0.3 # Probability to apply

    ##############################################
    # Random Resolution: low resolution
    r_low_resolution = True # Indicator if low resolution should be applied
    r_lr_scale_range = (0.5, 1) # Scale of each dimension is a random multiplier sampled from the given range
    r_lr_synchronize_channels=False # Same multiplier per channels
    r_lr_synchronize_axes=True # Same multiplier per each axes
    r_lr_probability_per_channels=0.5 # Probability to apply on each channel
    r_lr_probability = 0.25 # Probability to be applied

    ##############################################
    # Center crop
    c_center_crop = True # Indicator if center crop should be applied (necessary if rotation is applied)
    c_cc_desired_patch_shape = (128,128,128) # Tupple of three integers representing desired patch size
    c_cc_probability = 1 # Probability to be applied

    ##############################################
    # Remove label transform
    r_remove_label_transform = True # Indicator if we want to swap the label   
    r_rlt_source_label = -1 # Source label (integer) which is going to be swapped with destination label
    r_rlt_destination_label = 0 # Destination label (integer) label which will replace the source label
    r_rlt_probability = 1 # Probability to apply

    ##############################################
    # Segmentation mask resizing(Downsampling) for deep survesion
    s_ds_resize = True #Indicator if resizing should be applied
    s_ds_shapes_list = [[128, 128, 128], [64, 64, 64]] # List of shapes to which input seg mask will be resized
    s_ds_probability = 1 # Probability to be applied

    def __str__(self):
        '''
        Just to check params
        '''
        _retval = ''
        _retval += ("######################################\n")
        _retval += ("Augmentations")
        _retval += (f'name: {AugmentationConfigNNUnet.name}\n' +
                    f'rotation: {AugmentationConfigNNUnet.a_rotation}\n' + 
                    f'scaling: {AugmentationConfigNNUnet.a_scaling}\n' + 
                    f'mirror: {AugmentationConfigNNUnet.a_mirror}\n' + 
                    f'noise: {AugmentationConfigNNUnet.g_noise}\n' + 
                    f'blur: {AugmentationConfigNNUnet.g_blur}\n' + 
                    f'brightness: {AugmentationConfigNNUnet.i_brightness}\n' + 
                    f'contrast: {AugmentationConfigNNUnet.i_contrast}\n' +                     
                    f'gamma_1: {AugmentationConfigNNUnet.i_gamma_1}\n' + 
                    f'gamma_2: {AugmentationConfigNNUnet.i_gamma_2}\n' + 
                    f'low_resolution: {AugmentationConfigNNUnet.r_low_resolution}\n' +
                    f'resize: {AugmentationConfigNNUnet.s_ds_resize}\n' +            
                    f'center_crop: {AugmentationConfigNNUnet.c_center_crop}\n' +
                    f'remove_label_transform: {AugmentationConfigNNUnet.r_remove_label_transform}\n'
                    )
        _retval += ("######################################\n")
        return _retval
    
    def export_to_json(
            self, 
            export_path: str = None):
        """
        Exports the augmentation configuration to a JSON file.
        If export_path is a directory (i.e. does not end with ".json"),
        the filename "AugmentationConfigNNUnet_fingerprint.json" is appended.

        Args:
            export_path (str): Path where the config will be stored.
        """
        # Build the output dictionary using the class attributes.
        _output_dict = {
            "name": AugmentationConfigNNUnet.name,
            "a_rotation": AugmentationConfigNNUnet.a_rotation,
            "a_r_degrees": AugmentationConfigNNUnet.a_r_degrees,
            "a_r_probability": AugmentationConfigNNUnet.a_r_probability,
            "a_scaling": AugmentationConfigNNUnet.a_scaling,
            "a_s_scale": AugmentationConfigNNUnet.a_s_scale,
            "a_s_same_ratio": AugmentationConfigNNUnet.a_s_same_ratio,
            "a_s_probability": AugmentationConfigNNUnet.a_s_probability,
            "a_mirror": AugmentationConfigNNUnet.a_mirror,
            "a_m_axes": AugmentationConfigNNUnet.a_m_axes,
            "a_m_probabilty_per_axis": AugmentationConfigNNUnet.a_m_probabilty_per_axis,
            "a_m_probability": AugmentationConfigNNUnet.a_m_probability,
            "g_noise": AugmentationConfigNNUnet.g_noise,
            "g_n_mean": AugmentationConfigNNUnet.g_n_mean,
            "g_n_std": AugmentationConfigNNUnet.g_n_std,
            "g_n_probability": AugmentationConfigNNUnet.g_n_probability,
            "g_blur": AugmentationConfigNNUnet.g_blur,
            "g_b_sigma_range": AugmentationConfigNNUnet.g_b_sigma_range,
            "g_b_probability_per_channel": AugmentationConfigNNUnet.g_b_probability_per_channel,
            "g_b_synchronize_channels": AugmentationConfigNNUnet.g_b_synchronize_channels,
            "g_b_synchronize_axes": AugmentationConfigNNUnet.g_b_synchronize_axes,
            "g_b_probability": AugmentationConfigNNUnet.g_b_probability,
            "i_brightness": AugmentationConfigNNUnet.i_brightness,
            "i_b_intensity_range": AugmentationConfigNNUnet.i_b_intensity_range,
            "i_b_synchronize_channels": AugmentationConfigNNUnet.i_b_synchronize_channels,
            "i_b_probability_per_channels": AugmentationConfigNNUnet.i_b_probability_per_channels,
            "i_b_probability": AugmentationConfigNNUnet.i_b_probability,
            "i_contrast": AugmentationConfigNNUnet.i_contrast,
            "i_c_contrast_range": AugmentationConfigNNUnet.i_c_contrast_range,
            "i_c_preserve_range": AugmentationConfigNNUnet.i_c_preserve_range,
            "i_c_synchronize_channels": AugmentationConfigNNUnet.i_c_synchronize_channels,
            "i_c_probability_per_channels": AugmentationConfigNNUnet.i_c_probability_per_channels,
            "i_c_probability": AugmentationConfigNNUnet.i_c_probability,
            "i_gamma_1": AugmentationConfigNNUnet.i_gamma_1,
            "i_g_1_gamma_scale": AugmentationConfigNNUnet.i_g_1_gamma_scale,
            "i_g_1_invert_image": AugmentationConfigNNUnet.i_g_1_invert_image,
            "i_g_1_synchronize_channels": AugmentationConfigNNUnet.i_g_1_synchronize_channels,
            "i_g_1_probability_per_channels": AugmentationConfigNNUnet.i_g_1_probability_per_channels,
            "i_g_1_retain_stats": AugmentationConfigNNUnet.i_g_1_retain_stats,
            "i_g_1_probability": AugmentationConfigNNUnet.i_g_1_probability,
            "i_gamma_2": AugmentationConfigNNUnet.i_gamma_2,
            "i_g_2_gamma_scale": AugmentationConfigNNUnet.i_g_2_gamma_scale,
            "i_g_2_invert_image": AugmentationConfigNNUnet.i_g_2_invert_image,
            "i_g_2_synchronize_channels": AugmentationConfigNNUnet.i_g_2_synchronize_channels,
            "i_g_2_probability_per_channels": AugmentationConfigNNUnet.i_g_2_probability_per_channels,
            "i_g_2_retain_stats": AugmentationConfigNNUnet.i_g_2_retain_stats,
            "i_g_2_probability": AugmentationConfigNNUnet.i_g_2_probability,
            "r_low_resolution": AugmentationConfigNNUnet.r_low_resolution,
            "r_lr_scale_range": AugmentationConfigNNUnet.r_lr_scale_range,
            "r_lr_synchronize_channels": AugmentationConfigNNUnet.r_lr_synchronize_channels,
            "r_lr_synchronize_axes": AugmentationConfigNNUnet.r_lr_synchronize_axes,
            "r_lr_probability_per_channels": AugmentationConfigNNUnet.r_lr_probability_per_channels,
            "r_lr_probability": AugmentationConfigNNUnet.r_lr_probability,
            "c_center_crop": AugmentationConfigNNUnet.c_center_crop,
            "c_cc_desired_patch_shape": AugmentationConfigNNUnet.c_cc_desired_patch_shape,
            "c_cc_probability": AugmentationConfigNNUnet.c_cc_probability,
            "r_remove_label_transform": AugmentationConfigNNUnet.r_remove_label_transform,
            "r_rlt_source_label": AugmentationConfigNNUnet.r_rlt_source_label,
            "r_rlt_destination_label": AugmentationConfigNNUnet.r_rlt_destination_label,
            "r_rlt_probability": AugmentationConfigNNUnet.r_rlt_probability,
            "s_ds_resize": AugmentationConfigNNUnet.s_ds_resize,
            "s_ds_shapes_list": AugmentationConfigNNUnet.s_ds_shapes_list,
            "s_ds_probability": AugmentationConfigNNUnet.s_ds_probability,
        }

        if export_path is not None:
            # If export_path does not end with .json, assume it's a directory
            if not export_path.endswith(".json"):
                assert os.path.exists(export_path), f"Path for the export does not exist. Please check path: {export_path}!"
                export_path = os.path.join(export_path, "AugmentationConfigNNUnet_fingerprint.json")
            # Dump the dictionary to JSON
            with open(export_path, "w") as _json_file:
                json.dump(_output_dict, _json_file, indent=4)
        else:
            print("export_path is None. Dict cannot be exported!")

    def import_from_json(
            self, 
            import_path: str = None):
        """
        Imports configuration from a JSON file and updates the class attributes.

        Args:
            import_path (str): Path to the JSON file exported by export_to_json.
        """
        assert import_path is not None, "import_path must be provided"
        assert os.path.exists(import_path), f"Path for the import does not exist. Please check path: {import_path}!"

        with open(import_path, 'r') as _file:
            _imported_data = json.load(_file)

        # Update the class attributes with the imported data.
        AugmentationConfigNNUnet.name = _imported_data["name"]
        AugmentationConfigNNUnet.a_rotation = _imported_data["a_rotation"]
        AugmentationConfigNNUnet.a_r_degrees = _imported_data["a_r_degrees"]
        AugmentationConfigNNUnet.a_r_probability = _imported_data["a_r_probability"]
        AugmentationConfigNNUnet.a_scaling = _imported_data["a_scaling"]
        AugmentationConfigNNUnet.a_s_scale = _imported_data["a_s_scale"]
        AugmentationConfigNNUnet.a_s_same_ratio = _imported_data["a_s_same_ratio"]
        AugmentationConfigNNUnet.a_s_probability = _imported_data["a_s_probability"]
        AugmentationConfigNNUnet.a_mirror = _imported_data["a_mirror"]
        AugmentationConfigNNUnet.a_m_axes = _imported_data["a_m_axes"]
        AugmentationConfigNNUnet.a_m_probabilty_per_axis = _imported_data["a_m_probabilty_per_axis"]
        AugmentationConfigNNUnet.a_m_probability = _imported_data["a_m_probability"]
        AugmentationConfigNNUnet.g_noise = _imported_data["g_noise"]
        AugmentationConfigNNUnet.g_n_mean = _imported_data["g_n_mean"]
        AugmentationConfigNNUnet.g_n_std = _imported_data["g_n_std"]
        AugmentationConfigNNUnet.g_n_probability = _imported_data["g_n_probability"]
        AugmentationConfigNNUnet.g_blur = _imported_data["g_blur"]
        AugmentationConfigNNUnet.g_b_sigma_range = _imported_data["g_b_sigma_range"]
        AugmentationConfigNNUnet.g_b_probability_per_channel = _imported_data["g_b_probability_per_channel"]
        AugmentationConfigNNUnet.g_b_synchronize_channels = _imported_data["g_b_synchronize_channels"]
        AugmentationConfigNNUnet.g_b_synchronize_axes = _imported_data["g_b_synchronize_axes"]
        AugmentationConfigNNUnet.g_b_probability = _imported_data["g_b_probability"]
        AugmentationConfigNNUnet.i_brightness = _imported_data["i_brightness"]
        AugmentationConfigNNUnet.i_b_intensity_range = _imported_data["i_b_intensity_range"]
        AugmentationConfigNNUnet.i_b_synchronize_channels = _imported_data["i_b_synchronize_channels"]
        AugmentationConfigNNUnet.i_b_probability_per_channels = _imported_data["i_b_probability_per_channels"]
        AugmentationConfigNNUnet.i_b_probability = _imported_data["i_b_probability"]
        AugmentationConfigNNUnet.i_contrast = _imported_data["i_contrast"]
        AugmentationConfigNNUnet.i_c_contrast_range = _imported_data["i_c_contrast_range"]
        AugmentationConfigNNUnet.i_c_preserve_range = _imported_data["i_c_preserve_range"]
        AugmentationConfigNNUnet.i_c_synchronize_channels = _imported_data["i_c_synchronize_channels"]
        AugmentationConfigNNUnet.i_c_probability_per_channels = _imported_data["i_c_probability_per_channels"]
        AugmentationConfigNNUnet.i_c_probability = _imported_data["i_c_probability"]
        AugmentationConfigNNUnet.i_gamma_1 = _imported_data["i_gamma_1"]
        AugmentationConfigNNUnet.i_g_1_gamma_scale = _imported_data["i_g_1_gamma_scale"]
        AugmentationConfigNNUnet.i_g_1_invert_image = _imported_data["i_g_1_invert_image"]
        AugmentationConfigNNUnet.i_g_1_synchronize_channels = _imported_data["i_g_1_synchronize_channels"]
        AugmentationConfigNNUnet.i_g_1_probability_per_channels = _imported_data["i_g_1_probability_per_channels"]
        AugmentationConfigNNUnet.i_g_1_retain_stats = _imported_data["i_g_1_retain_stats"]
        AugmentationConfigNNUnet.i_g_1_probability = _imported_data["i_g_1_probability"]
        AugmentationConfigNNUnet.i_gamma_2 = _imported_data["i_gamma_2"]
        AugmentationConfigNNUnet.i_g_2_gamma_scale = _imported_data["i_g_2_gamma_scale"]
        AugmentationConfigNNUnet.i_g_2_invert_image = _imported_data["i_g_2_invert_image"]
        AugmentationConfigNNUnet.i_g_2_synchronize_channels = _imported_data["i_g_2_synchronize_channels"]
        AugmentationConfigNNUnet.i_g_2_probability_per_channels = _imported_data["i_g_2_probability_per_channels"]
        AugmentationConfigNNUnet.i_g_2_retain_stats = _imported_data["i_g_2_retain_stats"]
        AugmentationConfigNNUnet.i_g_2_probability = _imported_data["i_g_2_probability"]
        AugmentationConfigNNUnet.r_low_resolution = _imported_data["r_low_resolution"]
        AugmentationConfigNNUnet.r_lr_scale_range = _imported_data["r_lr_scale_range"]
        AugmentationConfigNNUnet.r_lr_synchronize_channels = _imported_data["r_lr_synchronize_channels"]
        AugmentationConfigNNUnet.r_lr_synchronize_axes = _imported_data["r_lr_synchronize_axes"]
        AugmentationConfigNNUnet.r_lr_probability_per_channels = _imported_data["r_lr_probability_per_channels"]
        AugmentationConfigNNUnet.r_lr_probability = _imported_data["r_lr_probability"]
        AugmentationConfigNNUnet.c_center_crop = _imported_data["c_center_crop"]
        AugmentationConfigNNUnet.c_cc_desired_patch_shape = _imported_data["c_cc_desired_patch_shape"]
        AugmentationConfigNNUnet.c_cc_probability = _imported_data["c_cc_probability"]
        AugmentationConfigNNUnet.r_remove_label_transform = _imported_data["r_remove_label_transform"]
        AugmentationConfigNNUnet.r_rlt_source_label = _imported_data["r_rlt_source_label"]
        AugmentationConfigNNUnet.r_rlt_destination_label = _imported_data["r_rlt_destination_label"]
        AugmentationConfigNNUnet.r_rlt_probability = _imported_data["r_rlt_probability"]
        AugmentationConfigNNUnet.s_ds_resize = _imported_data["s_ds_resize"]
        AugmentationConfigNNUnet.s_ds_shapes_list = _imported_data["s_ds_shapes_list"]
        AugmentationConfigNNUnet.s_ds_probability = _imported_data["s_ds_probability"]
    