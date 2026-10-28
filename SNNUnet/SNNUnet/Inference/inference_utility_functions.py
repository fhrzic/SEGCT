""" Utility functions for inference"""
import math
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MultipleLocator

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from DatasetPreprocessing.Utils.normalization import *
from DatasetPreprocessing.Utils.cropping_nnUnet import *
from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json

import SimpleITK as sitk
from einops import rearrange
from scipy.ndimage import gaussian_filter
from typing import Tuple

#**********************************************************************************************#
# METHODS HERE ARE WRITTEN BASED ON OTHER METHODS THAT CAN BE SEEN IN SCRIPTS BUT ARE KEPT
# SEPARATED SO THE CONSISTENCY IS KEPT CLEAR
#**********************************************************************************************#

def symetrical_padding(
    volume: torch.tensor = None, 
    desired_shape: tuple = None, 
    patch_shape: tuple = None,
    divisible_by_patch: bool = False,
    padding_value: int = 0) -> Tuple[torch.tensor, tuple, tuple]:
    """
    Method which adds padding symetrically (symetrical as possible) before and after the volume 
    until it is fully devisible or the desired_shape is reached. If the devisable_by_patch is 
    True, then it is padded more until it is devisable by patch.

    Args:
        * volume, torch.tensor, input volume which must be padded
        * desired_shape, tuple, shape which we want it to be. NOW, if the original volume shape 
        is in any dimension smaller than desired_shape, it is symetrically zero padded to it. 
        If it is bigger, it is left as it is.
        * patch_shape, tuple, shape of the patch wich is usefull if divisible_by_patch is turned on
        * divisiable_by_patch, bool, if set True, the volume is padded even more until in 
        every dimnesion it is divisable by corresponding dimension in the patch_shape
        * padding_value, int, values with witch the tensor will be padded.

    Returns:
        * torch.tensor, padded torch tensor
        * tuple, the offset from (0,0,0) which allows reconstruction of the original shape
        * tuple, (x,y,z) original volume shape for reconstruction purposes.
    """
    # Create containers
    _total_padding_x = _total_padding_y = _total_padding_z = 0

    # Obtain original volume shape
    _original_x, _original_y, _original_z = volume.shape

    # Make estimated volume
    _estimated_x, _estimated_y, _estimated_z = volume.shape

    # Obtain patch sizes
    _patch_x, _patch_y, _patch_z = patch_shape

    # Solve if the volume is smaller of desired volume_shape
    _desired_x, _desired_y, _desired_z = desired_shape

    if _original_x < _desired_x:
        _total_padding_x += _desired_x - _original_x
        _estimated_x = _desired_x

    if _original_y < _desired_y:
        _total_padding_y += _desired_y - _original_y
        _estimated_y = _desired_y

    if _original_z < _desired_z:
        _total_padding_z += _desired_z - _original_z
        _estimated_z = _desired_z

    # Solve divisible by patch
    if divisible_by_patch:
        if _estimated_x % _patch_x != 0:
            _total_padding_x += (_patch_x - (_estimated_x % _patch_x))

        if _estimated_y % _patch_y != 0:
            _total_padding_y += (_patch_y - (_estimated_y % _patch_y))

        if _estimated_z % _patch_z != 0:
            _total_padding_z += (_patch_z - (_estimated_z % _patch_z))

    # Calculate paddings
    _before_x, _after_x = _total_padding_x // 2 , _total_padding_x - (_total_padding_x // 2)
    _before_y, _after_y = _total_padding_y // 2 , _total_padding_y - (_total_padding_y // 2)
    _before_z, _after_z = _total_padding_z // 2 , _total_padding_z - (_total_padding_z // 2)
    # Pad torch tensor
    _padded_tensor = F.pad(volume, 
                           [_before_z, _after_z, _before_y, _after_y, _before_x, _after_x],
                           mode='constant', 
                           value = padding_value)
    # Return
    return [_padded_tensor,
            (_before_x, _before_y, _before_z),
            (_original_x, _original_y, _original_z)]

def unpad_volume(
    volume: torch.tensor = None, 
    padding_offset: tuple = None,
    original_volume_shape: tuple = None)->torch.tensor:
    """
    Method for unpadding the volume based on padding before and original_volume_shape.

    Args:
        * volume, torch.tensor, input volume which was previously padded
        * padding_beffore, tuple, x,y,z offset from 0 where the real content of the original
        volume starts.
        * original_volume_shape, tuple, tuple defining preserved volume shape
    """
    # Split
    _start_x, _start_y, _start_z = padding_offset
    _end_x, _end_y, _end_z = original_volume_shape

    # Fix paddings
    _end_x += _start_x
    _end_y += _start_y
    _end_z += _start_z

    # Return
    return volume[_start_x:_end_x, _start_y:_end_y, _start_z:_end_z]

def split_volume_into_patches_no_padding(
    volume: torch.tensor = None,
    patch_shape: tuple = None,
    overlap: float = 0)->torch.tensor:
    """
    Split the 3D volume into (non)overlapping patches of shape self.patch_shape. But instead of zero
    padding the volume to the closes devisable shape, it does reverse. It creates all normal 
    patches, then it makes patches on the residuals. E.G. On 1D example let's say that volume is 
    130px long and patch is 100. It will make two patches frist one from 0-100, and second from 
    30-100. It will not zero pad the volume to 200 and then make two patches: from 0-100 and 
    100-200.

    The overlap of the same pixels is handeled the same way as the overlapping is included.

    Args:
        * volume, torch.tensor, volume to be splitted
        * patch_shape, tuple, size of the patch given as (x, y, z)
        * overlap, float, percetange of the overlap between patches

    Returns:
        * patches, torch tensor conataining patchs
        * patches_locations, list, list of where the patches beggining coordinates are located.

    """
    # First we deal with regular patches
    # Get volume shape
    _x, _y, _z = volume.shape

    # Obtain patch shape
    _patch_x, _patch_y, _patch_z = patch_shape

    # Obtain stride
    _stride_x, _stride_y, _stride_z = [int((1 - overlap) * _size) for _size in patch_shape]

    # Get the regular patch locations from the unfolding process
    _patch_locations = []
    _patches = []

    # Get positions along _x, _y and _z
    _x_positions = list(range(0, _x - _patch_x + 1, _stride_x))
    _y_positions = list(range(0, _y - _patch_y + 1, _stride_y))
    _z_positions = list(range(0, _z - _patch_z + 1, _stride_z))

    # Handle the residual space at the end of the volume
    if (_x - _patch_x) % _stride_x != 0:
        _x_positions.append(_x - _patch_x)
    if (_y - _patch_y) % _stride_y != 0:
        _y_positions.append(_y - _patch_y)
    if (_z - _patch_z) % _stride_z != 0:
        _z_positions.append(_z - _patch_z)

    for _i in _x_positions:
        for _j in _y_positions:
            for _k in _z_positions:
                # Calculate locations
                _start_x = _i 
                _start_y = _j 
                _start_z = _k 
                _end_x = _start_x + _patch_x
                _end_y = _start_y + _patch_y
                _end_z = _start_z + _patch_z
                # Update
                _patch_locations.append((_start_x, _start_y, _start_z))

                # Update volume patches
                _patches.append(volume[_start_x :_end_x,
                                       _start_y : _end_y,
                                       _start_z : _end_z])
    _patches = torch.stack(_patches, dim = 0)
    return _patches, _patch_locations

def reconstruct_volume_from_patches(
    patches: torch.Tensor = None,
    patch_locations: list = None,
    volume_shape: tuple = None,
    patch_shape: tuple = None,
    gaussian: bool = False,
    number_of_classes: int = 7,
    device: str = "cpu")->torch.Tensor:
    """
    Reconstruct the 3D volume from patches by averaging overlapping regions.
    
    Args:
        * patches: torch.tensor, Tensor containing the patches
        * patch_locations: list, List of patch start coordinates (x, y, z)
        * volume_shape: tuple, Shape of the original volume (x, y, z)
        * patch_shape: tuple, Shape of each patch (x, y, z)
        * gaussian, bool, if true, gaussian weightening will be applied over 
        each patch
        * number_of_classes, int, number of classes in the segmentation map
        * device, str, place where are the tensor will be set. 
    
    Returns:
        * reconstructed_volume (torch.tensor): The reconstructed 3D volume
    """
    # Transfer to device (evede OOM)
    try:
        _patches = patches.to(device)
        # Initialize an empty tensor for reconstruction      
        _reconstructed_volume = torch.zeros((number_of_classes,) + volume_shape, 
                                            dtype=torch.float16).to(_patches.device)
    except RuntimeError:
        _patches = patches.to("cpu")
        # Initialize an empty tensor for reconstruction  
        _reconstructed_volume = torch.zeros((number_of_classes,) + volume_shape, 
                                            dtype=torch.float16).to("cpu")

    # Obtain gaussian if necessary
    if gaussian:
        # Get gaussian
        _gaussian = compute_gaussian(patch_shape = patch_shape,
                                     device = _patches.device)
        # Solve the dimension problem
        _gaussian = _gaussian.unsqueeze(0)
    else:
        _gaussian = 1

    # Loop over each patch and its corresponding location
    for _patch, (_start_x, _start_y, _start_z) in zip(_patches, patch_locations):
        # Apply gaussian weightening (it is 1 if gaussian is not applied)
        _patch *= _gaussian
        # Place the patch back into the reconstructed volume
        _reconstructed_volume[
            :,
            _start_x:_start_x + patch_shape[0],
            _start_y:_start_y + patch_shape[1],
            _start_z:_start_z + patch_shape[2]
        ] += _patch

    # Release memory
    del _patches
    torch.cuda.empty_cache()

    # Create counter tensor (I know, this could all be done in one for Loop -
    # but it can easily go OOM on GPU for big tensors)
    try:
        _overlap_counter = torch.zeros((number_of_classes,) + volume_shape, 
                                       dtype=torch.float16).to(_reconstructed_volume.device)
    except RuntimeError:
        _overlap_counter = torch.zeros((number_of_classes,) + volume_shape, 
                                       dtype=torch.float16).to("cpu")
        _reconstructed_volume = _reconstructed_volume.to("cpu")
        _gaussian = _gaussian.to("cpu")

    for (_start_x, _start_y, _start_z) in patch_locations:
        # Increment the overlap counter to keep track of how many times each voxel is covered
        _overlap_counter[
            :,
            _start_x:_start_x + patch_shape[0],
            _start_y:_start_y + patch_shape[1],
            _start_z:_start_z + patch_shape[2]
        ] += _gaussian

    # Avoid division by zero for areas not covered by any patch
    try:
        _overlap_counter[_overlap_counter == 0] = 1
    except RuntimeError:
        _overlap_counter = _overlap_counter.to("cpu")
        _overlap_counter[_overlap_counter == 0] = 1
        _overlap_counter = _overlap_counter.to(_reconstructed_volume)

    # Average the overlapping regions
    try:
        _reconstructed_volume /= _overlap_counter
    except RuntimeError:
        _reconstructed_volume = _reconstructed_volume.to("cpu")
        _overlap_counter = _overlap_counter.to("cpu")
        _reconstructed_volume /= _overlap_counter
        _reconstructed_volume = _reconstructed_volume.to(device)

    return _reconstructed_volume

def compute_gaussian(
    patch_shape: tuple = None,
    sigma_scale: float = 1./8,
    value_scaling_factor: float = 10.0,
    dtype = torch.float16, device: str = "cpu")->torch.tensor:
    """
    Function which calculates gaussian matrix over the given patch. Gaussian's peak
    is located at the center of the patch.
    
    Args:
        * patch_shape, tuple, shape of the torch patch to export
        * sigma_scale, float, number controling spread of gaussian. This is inherited
        from nnUnet as value.
        * value_scaling_factor, float, max value of the gausian - it is scaling factor. 
        Also inherited from nnUnet as value.
        * dtype, data type of the export tensor.
        * device, string, device where to place the output tensor
    
    Returns:
        * torch_tensor: tensor representng gaussian coeficients accross the given patch size
    """
    # Create temporary storage
    _tmp_storage = np.zeros(patch_shape)

    # Obtain center cordinate
    center_coords = [_i // 2 for _i in patch_shape]

    # Calculate sigmas
    _sigmas = [_i * sigma_scale for _i in patch_shape]

    # Set center to max value
    _tmp_storage[tuple(center_coords)] = 1

    # Calcualte map
    _gaussian_importance_map = gaussian_filter(_tmp_storage, _sigmas, 0, mode='constant', cval=0)

    _gaussian_importance_map = torch.from_numpy(_gaussian_importance_map)

    # Scale it to 0-value_scaling_factor
    _gaussian_importance_map /= (torch.max(_gaussian_importance_map) / value_scaling_factor)

    # Send it to device
    _gaussian_importance_map = _gaussian_importance_map.to(device=device, dtype=dtype)

    # Gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    _mask = _gaussian_importance_map == 0

    # Try to calculate it on GPU, if not possible, CPU it is
    try:
        _gaussian_importance_map[_mask] = torch.min(_gaussian_importance_map[~_mask])
    except RuntimeError:
        _gaussian_importance_map = _gaussian_importance_map.to('cpu')
        _mask = _mask.to('cpu')
        _gaussian_importance_map[_mask] = torch.min(_gaussian_importance_map[~_mask])
        _gaussian_importance_map = _gaussian_importance_map.to(device)


    return _gaussian_importance_map

def transfer_ct_to_npy(
    path_to_ct: str = None, 
    normalization_fingerprint_path: str = None,
    data_fingerprint_path: str = None)->np.array:
    """
    Function which transfers ct to npy based on proveded data_fingerprint json.

    Args:
        * path_to_ct, str, path to the ct which needs to be converted
        * normalization_fingerprint_path, str, path to the fingerprint json which is generated by scripts
        in dataset_formating.py/dataset_export.py scripts (normalization_fingerprint.json).
        * data_fingerprint_path, str, path to the data fingerprint json generated by
        scripts in DatasetFingerprint/generate_dataset_fingerprint.py. This is not necessary unless
        implementation method is not F.
    
    Returns:
        * np.array which represents normalized version of given CT
    """
    # Check for files
    assert path_to_ct is not None, \
        "Required files for transfer_to_npy are not provided!"

    # Read Volume
    _input_volume = sitk.ReadImage(path_to_ct)
    _volume_numpy_array = sitk.GetArrayFromImage(_input_volume)    
    _volume_numpy_array = rearrange(_volume_numpy_array, 'z y x -> x y z')

    # Obtain normalizing parameters
    if normalization_fingerprint_path is not None:
        # Read normalization json
        with open(normalization_fingerprint_path, 'r', encoding='utf-8') as _f:
            _data = json.load(_f)
            _normalization_method = _data["normalization_method"]
        ### Obtain normalization
        assert _normalization_method in ["F", "V", "M"], \
           "Error: Invalid normalization selection." + \
           f"Implemented normalizations methods are 'V', 'F', and 'M'! Got {_normalization_method}!"
        if _normalization_method == "F":
            _CT_normalization = NormalizeCTFromFingerprint(
                path_to_fingerprint = data_fingerprint_path)
        
        if _normalization_method == "V":
            _CT_normalization = NormalizeCTFromVolume()
            
        if _normalization_method == "M":
            _CT_normalization = NormalizeCTFromMask()

        # Perform normalization for CT
        _volume_numpy_normalized = _CT_normalization.run(_volume_numpy_array) #_cropped_volume

        # Adjust type
        _volume_numpy_normalized = _volume_numpy_normalized.astype(np.float32)

        # Return
        return _volume_numpy_normalized
    else:
        # No normalization (usefull in case of loading segmentation mask)
        return _volume_numpy_array #_cropped_volume

def transfer_to_sitkImg(
    data: np.ndarray = None, 
    original_ct_path: str = None)->sitk.Image:
    """
    Simple script which transfers numpy array to sitk.Image

    Args:
        * data, np.ndarray, data which must be transered
        * original_ct_path, str,  path to original file necessary to obtain properties 
        of the volume so that data can be appropriately transfered to nii.gz
    
    Returns:
        * sitk.Image representing nii.gz version of provided np.ndarray
    """
    # Load the NIfTI file using SimpleITK
    _original_image = sitk.ReadImage(original_ct_path)

    # Extract the metadata
    _origin = _original_image.GetOrigin()
    _spacing = _original_image.GetSpacing()
    _direction = _original_image.GetDirection()

    # Obtain sitk data
    _numpy_array_data = rearrange(data, 'z y x -> x y z')
    _sitk_data = sitk.GetImageFromArray(_numpy_array_data)

    # Set the metadata to match the original image
    _sitk_data.SetOrigin(_origin)
    _sitk_data.SetSpacing(_spacing)
    _sitk_data.SetDirection(_direction)   

    # Return
    return _sitk_data


def obtain_statistics(
    gt_segmentation_path: str = None,
    pred_segmentation_path: str = None,
    export_path: str = None)->dict:
    """
    Function which obtains statistics from provided gt_segmentation_path and 
    predicted_segmentation. Segmentation can be either parth to npy or path to ct. 
    It is determined by extension [".nii.gz", or ".npy"].

    Args:
        * gt_segmentation_path, str, path to the ground truth segmentation [".nii.gz", or ".npy"].
        * pred_segmentation_path, str, path to the predicted segmentation [".nii.gz", or ".npy"].
        * path, str, path to where "name.json" contatining exporting dictionary will be stored. The 
         name is derived from gt_segmentation_path.

    Returns:
        * Dictionary of measurred coefficients per class detected
    """
    # Loading the data
    # Check for file extension validity
    _file_name_full = os.path.basename(gt_segmentation_path)
    if "nii.gz" in _file_name_full:
        _file_extension = "nii.gz"
    if "npy" in _file_name_full:
        _file_extension = "npy"

    assert _file_extension in ["npy", "nii.gz"], \
        f"Wrong extension! Expected .npy or .nii.gz but got {_file_extension}!"
    _file_name = _file_name_full.split(f".{_file_extension}")[0]

    # In case of numpy
    if _file_extension == "npy":
        # Load
        _gt_array = np.load(gt_segmentation_path)
        _pred_array = np.load(pred_segmentation_path)
    else:
        # In case of CT, extract npy
        _gt_array = transfer_ct_to_npy(path_to_ct = gt_segmentation_path,
                                        normalization_fingerprint_path = None)
        _pred_array = transfer_ct_to_npy(path_to_ct = pred_segmentation_path,
                                        normalization_fingerprint_path = None)

    # CALCULATING METRICS
    # Obtain classes
    _gt_classes = np.unique(_gt_array)
    _gt_classes = _gt_classes[_gt_classes != 0]
    _predicted_classes = np.unique(_pred_array)
    _predicted_classes = _predicted_classes[_predicted_classes != 0]
    # Export dict
    _export_dict = {}
    _overall_dict = {"tp": [],
                     "fp": [],
                     "fn": [],
                     "precision": [],
                     "recall": [],
                     "dice": [],
                     "f1_score": []}
    
    # I am skipping background
    for _cls in _predicted_classes:
        # Obtain masks
        _pred_mask = (_pred_array == _cls)
        _label_mask = (_gt_array == _cls)

        # Get calculations
        _tp = (_pred_mask & _label_mask).astype(float).sum()
        _fp = (_pred_mask & ~_label_mask).astype(float).sum()
        _fn = (~_pred_mask & _label_mask).astype(float).sum()

        # Update overall dict
        _overall_dict["tp"].append(_tp)
        _overall_dict["fp"].append(_fp)
        _overall_dict["fn"].append(_fn)

        # If cls is not in GT we are skipping it
        if _cls not in _gt_classes:
            continue
        
        # Calculate precision recall, f1score and dice    
        _precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else float('nan')
        _recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else float('nan')
        _f1_score = 2*(_precision*_recall) / (_precision + _recall) if _precision+_recall > 0 else 0.0
        _dice = (2*_tp) / (2 *_tp + _fp + _fn) if (2* _tp + _fp + _fn) > 0 else 1.0

        # Generate measurments for this class
        _meassurment_dict = {"tp": _tp,
                             "fp": _fp,
                             "fn": _fn,
                             "precision": _precision,
                             "recall": _recall,
                             "dice": _dice,
                             "f1_score": _f1_score}

        # Update dict
        if math.isnan(_precision) == False: 
            _overall_dict["precision"].append(_precision)
        if math.isnan(_recall) == False: 
            _overall_dict["recall"].append(_recall)
        if math.isnan(_f1_score) == False: 
            _overall_dict["f1_score"].append(_f1_score)
        if math.isnan(_dice) == False: 
            _overall_dict["dice"].append(_dice)

        # Update class dict
        _export_dict[float(_cls)] = _meassurment_dict

    # Calculate micro precission, recall and f1 score (MICRO)
    _overall_tp = np.sum(_overall_dict["tp"])
    _overall_fp = np.sum(_overall_dict["fp"])
    _overall_fn = np.sum(_overall_dict["fn"])
    
    _precision = _overall_tp / (_overall_tp + _overall_fp) if (_overall_tp + _overall_fp) > 0 else float('nan')
    _recall = _overall_tp / (_overall_tp + _overall_fn) if (_overall_tp + _overall_fn) > 0 else float('nan')
    _f1_score = 2*(_precision*_recall) / (_precision + _recall) if _precision+_recall > 0 else 0.0
    _dice = (2*_overall_tp) / (2 *_overall_tp + _overall_fp + _overall_fn) if (2* _overall_tp + _overall_fp + _overall_fn) > 0 else 1.0

    _micro_results_dict = {}
    _micro_results_dict["tp"] = _overall_tp
    _micro_results_dict["fp"] = _overall_fp
    _micro_results_dict["fn"] = _overall_fn
    _micro_results_dict["precision"] = _precision
    _micro_results_dict["recall"] = _recall
    _micro_results_dict["dice"] = _dice
    _micro_results_dict["f1_score"] = _f1_score

    _export_dict["micro_overall"] = _micro_results_dict
    
    # Calculate mean precission, recall and f1 score (MACRO)
    for _key in _overall_dict:
        _overall_dict[_key] = np.mean(_overall_dict[_key])
    _export_dict["macro_overall"] = _overall_dict

    # Get name
    _name = os.path.join(export_path, f"{_file_name}.json")

    # Export
    with open(_name, 'w') as json_file:
        json.dump(_export_dict, json_file, indent=4)

def plot_metrics(
        output_path:str = None, 
        metrics_avg: dict = None):
    """
    Method for plotting metrices

    Args:
        * output_path, string, path to the place where the plot will be outputed
        * metric_avg, dict, dictionary containg metrics calcualted in the statistics_evaluation
    """
    # Filter to get only class metrics, ignoring the "overall" key
    _classes = [_key for _key in metrics_avg.keys() if _key not in ['macro_overall', 'micro_overall']]
    _metrics = ['precision', 'recall', 'f1_score', 'dice']

    # Prepare data for plotting
    _x = np.arange(len(_classes))  # the label locations
    _width = 0.2  # _width of each bar

    fig, _ax = plt.subplots(figsize=(12, 6))

    # Plot each metric as a group of bars
    for _i, _metric in enumerate(_metrics):
        _values = [metrics_avg[str(_cls)][_metric] for _cls in _classes]
        _ax.bar(_x + _i * _width, _values, _width, label=_metric.capitalize())

    # Add horizontal lines for the overall _metrics
    colors = {'precision': 'blue', 'recall': 'green', 'f1_score': 'orange', 'dice': 'red'}
    for _metric in _metrics:
        overall_value = metrics_avg['macro_overall'][_metric]
        _ax.axhline(y=overall_value, 
                    color=colors[_metric], 
                    linestyle='--', 
                    label=f'macro_overall {_metric.capitalize()}')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    _ax.set_xlabel('Class')
    _ax.set_ylabel('Score')
    _ax.set_title('Average Precision, Recall, F1 Score and Dice for Each Class')
    _ax.set_xticks(_x + _width)
    _ax.set_xticklabels(_classes)
    _ax.legend()

    # Add gridlines and additional y-axis ticks for readability
    _ax.yaxis.set_major_locator(plt.MaxNLocator(10))  # More y-axis ticks
    _ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "merged_metrics.png"))
    plt.show()

def merge_statistics(
    input_path: str = None, 
    path_to_fingerprint: str = None,
    output_path:str = None):
    """
    Main function for merging and exporting statistics

    Args:
        * input_path, str, path to the input dir which contains cases. It is partent 
        dir of all cases and it seeks for the files ending with .json
        * path_to_fingerprint, str, path to the fingerprint file
        * output_path, str, path to the folder where output statistics will be made
    """
    # Read fingerprint and obtain labels
    _fingerprint_data = read_data_fingerprint_json(path_to_fingerprint)
    _labels_names = _fingerprint_data["LABELS_MAPPING"]

    # Initialize a dictionary to accumulate the values
    _metrics_sum = defaultdict(lambda: defaultdict(list))

    # Iterate over each JSON file in the directory
    for _root, _, _files in os.walk(input_path):
        for _filename in _files:
            if _filename.endswith('.json') and _filename not in ["inference_config.json", "times.json"]:
                _file_path = os.path.join(_root, _filename)
                with open(_file_path, 'r') as _file:
                    _data = json.load(_file)

                # Aggregate values for each class and overall
                for _key, _value in _data.items():
                    for _metric, _metric_value in _value.items():
                        _metrics_sum[_key][_metric].append(_metric_value)
    # Calculate averages
    _metrics_avg = {}
    for _key, _metric_dict in _metrics_sum.items():
        try:
            _metrics_avg[_labels_names[str(int(float(_key)))]] = {
                _metric: np.mean(_values) for _metric, _values in _metric_dict.items()}
        except ValueError:
            _metrics_avg[_key] = {
                _metric: np.mean(_values) for _metric, _values in _metric_dict.items()}

    # Dump into json
    with open(os.path.join(output_path, "merged_metrics.json"), 'w') as _f:
        json.dump(_metrics_avg, _f, indent=4)

    # Plot
    plot_metrics(output_path = output_path, 
                 metrics_avg = _metrics_avg)


def plot_times(
    times_json_path: str = None,
    output_path: str = None):
    """
    Ploting times
    Args: 
        * times_json_path, str, directory where times.json is located
        * output_path, str, directory where the output will be stored
    """
    # Load the JSON file
    with open(times_json_path, 'r') as _json_file:
        _data = json.load(_json_file)

    # Extract filenames and values
    _filenames = list(_data.keys())
    _values = list(_data.values())

    # Calculate mean and standard deviation
    _mean_value = np.mean(_values)
    _std_value = np.std(_values)

    _x_labels = []
    for _i, _filepath in enumerate(_filenames):
        _file_size_gb = os.path.getsize(_filepath) / (1024 ** 3)  # Convert bytes to GB
        _x_labels.append(f"{_i+1}_{_file_size_gb:.1f}")

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(_x_labels, _values, color='skyblue', edgecolor='black')

    # Add mean and standard deviation lines
    plt.axhline(_mean_value, color='red', linestyle='--', label=f'Mean = {_mean_value:.2f}')
    plt.axhline(_mean_value + _std_value, 
                color='green', 
                linestyle='--', 
                label=f'Mean + STD = {_mean_value + _std_value:.2f}')
    plt.axhline(_mean_value - _std_value, 
                color='purple', 
                linestyle='--', 
                label=f'Mean - STD = {_mean_value - _std_value:.2f}')

    # Enable horizontal grid lines only and set y-axis ticks to increments of 5
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)
    _y_max = max(_values)
    _y_tick_step = max(1, int(_y_max / 10))  # Show ~10 y-ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(_y_tick_step))

    # Add labels and legend
    plt.xlabel('File Size (idx_GB)')
    # Set x-axis labels and positions
    plt.xticks(range(0, len(_values)), _x_labels, rotation=90, ha="right")
    plt.ylabel('Time(sec)')
    plt.title('Values per File with Mean and Std Deviation Lines')
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "times.png"))
    plt.show()
