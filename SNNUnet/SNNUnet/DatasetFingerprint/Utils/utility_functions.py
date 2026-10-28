""" Functions and classes for generating fingerprints"""
import os
import sys
import json
import multiprocessing
from collections import Counter
from einops import rearrange
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from DatasetPreprocessing.Utils.cropping_nnUnet import *


#**********************************************************************************#
# UTILS
def read_data_fingerprint_json(
    path_to_json: str = None):
    """
    Method which parses and return five dictionaries from the given json file exported by
    DataFingerprint class.

    Args:
        * path_to_json, str, path to the json exported by DataFingerprint.ExportFingerprint method

    Returns:
        * Returns dictionary which conatins five dictionaries each conatianing speciffic 
        properties about data. Please see ExportFingerprint method for in-detail description of 
        each dictionary (it is lenghty description).
    """

    # Initialize a dictionary to store the different sections
    _sections = {
        "GENERAL_INFO": None,
        "LABELS_MAPPING": None,
        "VOXEL_STATISTICS_SAMPLES": None,
        "VOXEL_STATISTICS_MEANS": None,
        "CLASS_STATISTICS": None
    }

    # Read the file as a string
    with open(path_to_json, 'r') as _file:
        _content = _file.read()

    # Split the file content based on the headers (remove the header lines)
    for _section in _sections:
        # Find the content between the section and the next header
        _start_idx = _content.find(_section) + len(_section)
        _next_header_idx = _content.find(
            "**********************************************************************", 
            _start_idx)

        # Extract the JSON portion for each section
        _json_content = _content[_start_idx:_next_header_idx].strip()

        # Remove any unnecessary headers or formatting
        _json_content = _json_content.strip().strip("{}").strip()

        # Add back curly braces to make it valid JSON for parsing
        _json_content = "{" + _json_content + "}"

        # Parse the JSON string into a dictionary and store it
        _sections[_section] = json.loads(_json_content)

    return _sections

#**********************************************************************************#
# CLASS
class DataFingerprint():
    """
    Class for handling data
    """
    def __init__(
        self,
        CT_files = None,
        seg_maps = None,
        file_extension: str = "nii.gz",
        labels = None,
        ignore_label: int = -1,
        number_of_processes: int = None):
        """
        Init function which serves to build up the class. IMPORTATNT IN CASE OF PROVIDING LISTS:
        The CT_files and seg_maps must be paired (1st file in CT_files must correspond to the 1st 
        file in seg_maps).
        
        Args:
            * CT_files, list or string, either list of file names, or string which represent path 
            to the CTs (directory conatining CTs)
            * seg_maps, list or string, either list of file names, or string which represent path 
            to the segmentation maps (directory conatining segmentation maps)
            * file_extension, string, extension of CT_files and seg_maps. Based on this extensions 
            the maps/files will be searched
            * labels, dict or string, either dict contatining "pixel_value": "label_name" pairs, 
            or a string representing path to the json file contatining the previously described dict
            * ignore_label, int, during the training, in cases when we need to zero pad the patch 
            because it exceeds boundaries of the input volume, we need to set
            ignore label in segmentation masks so when calculating the loss, the loss will not 
            take into ignore padded voxels. DISCLAIMER: NO REAL LABELS MUST BE NAMED "ignore_label", 
            and "ignore_label" value must not be among real labels.
            number_of_processes, int, if not None, then it uses the given number of processes when 
            processing data
        
        """

        ### Obtain CT_files
        # if it is list, just update self variable
        # if it is string, call method which seeks for nii.gz in given dir
        if isinstance(CT_files, list):
            self.CT_files = CT_files
        elif isinstance(CT_files, str):
            self.CT_files = self.__obtain_files(
                path_to_files = CT_files,
                file_extension = file_extension)
        else:
            raise ValueError("Variable CT_files must be either list or string")
        
        ### Obtain seg_maps
        # if it is list, just update self variable
        # if it is string, call method which seeks for nii.gz in given dir
        if isinstance(seg_maps, list):
            self.seg_maps = seg_maps
        elif isinstance(seg_maps, str):
            self.seg_maps = self.__obtain_files(
                path_to_files = seg_maps,
                file_extension = file_extension)
        else:
            raise ValueError("Variable seg_maps must be either list or string")

        ### Obtain label dictionary
        # If it is dict, just update self variable,
        # if it is string, then it will load the dict from the json to the dict
        if isinstance(labels, dict):
            self.labels = labels
        elif isinstance(labels, str):
            self.labels = self.__obtain_labels(path_to_labels = labels,
                                              ignore_label = ignore_label)

        ### Set number of processes
        self.number_of_processes = number_of_processes

        ### Caluclate number of fg voxels for intensity estimation. Following
        # NNUnet guide, we set it to be 10e7 // number of cases. 
        self.number_of_fg_voxels_to_evaluate = int(10e7 // len(self.seg_maps))

    def __obtain_files(
        self, 
        path_to_files:str = None, 
        file_extension: str = "nii.gz")->list:
        """
        Method which lists the files in path_to_files and extract all files which ends with 
        file_extension

        Args:
            * path_to_files, str, path to the files 
            * file_extension, str, extension of the files which will be addet to the list

        Returns:
            * list of files found in a given "path_to_files" with extension given by 
            "file_extension". The file list is sorted so it will match 
        """

        # Check if file exist
        assert os.path.exists(path_to_files), \
            f"Cant find: {path_to_files}! Please check the file path."

        # Check if it is directory
        assert os.path.isdir(path_to_files), \
            f"Not dir: {path_to_files}! Please check the file path is path to dir."

        # Obtain files
        _file_list = [os.path.join(path_to_files, _file) for
                      _file in os.listdir(path_to_files) if _file.endswith(file_extension)]

        # Sort file list
        _file_list.sort()

        # Return
        return _file_list

    def __obtain_labels(self, 
                       path_to_labels: str = None,
                       ignore_label: int = -1)->dict:
        """
        Method which lists the labels the given json file presented as path_to_labels. 
        Json file contains ditcs where one key is labels. That key points to the 
        previsously described pixel-->name pair

        Args:
            * path_to_labels, str, path to the json file containing labels
            * ignore_label, int, lables which will be used to pad masks 
            (input is padded by backgorund, segmentation masks by ignore label). This
            label will be ignored during lost calculation. So in the original 
            segmentation masks -1 must be avoided as a label.
        
        Returns:
            * dictionary (dict) contatining pairs "pixel_value" (int) --> "label_name" (str)
        """
        # Check if file exist
        assert os.path.exists(path_to_labels), \
            f"Cant find: {path_to_labels}! Please check the file path."

        # Check if it json
        assert path_to_labels.endswith(".json"), \
            f"Not json: {path_to_labels}! Please check the file path is path to json file."

        # Load json
        try:
            # Load the JSON file into a dictionary
            with open(path_to_labels, 'r') as _f:
                _data = json.load(_f)
        except RuntimeError as _exc:
            raise RuntimeError(f"Could not open {_f}. Check if it is a json") from _exc

        # Fix keys
        try:
            _data = _data["labels"]
            _data = {int(_key): _value for _key, _value in _data.items()}
        except ValueError as _exc:
            raise ValueError("Dictionary key must be integer!. "
                             "Failed in conversion (they represent pixel values)") from _exc

        assert "ignore_label" not in list(_data.values()), \
            ('"ignore_label" is a reserved word and such label must not exists among real labels!.'
            'Please change real label name to something else and avoid having -1 as a value in'
            'segmentation masks!')
        assert ignore_label not in list(_data.keys()), \
            (f"Unofrtunatley {ignore_label} value set for ignore_label already exists among real"
             "labels. Please change the ignore_label value!")
        _data[ignore_label] = "ignore_label"

        # Return
        return _data

    def _analyise_one_case(
        self, ct_data_path: str = None, 
        seg_mask_path: str = None, 
        number_of_voxels: int = 10000)->dict:
        """
        Method which is a "middleman" for reading and extracting one volume. It returns all data 
        for that volume. This is necessary for multiprocesing.

        Args:
            * ct_path_data, str, path to the ct volume
            * seg_mask_paht, str, path to the segmentation mask 
            * number_of_voxels, int, number of random voxels that will be sampled from the sample 
            (for later statistics).
        
        Returns:
            * dictionary with all propertis of a given volume/segmentation mask.
            All values are calculated on cropped image (image reduced to its foreground/classes 
            containng bbox)
            Properties:
            --> spacing, tupple, three numbers (floats) representing spacing in each direction
            --> number_of_channels, int, one number representing number of channels in volume
            --> z_channel_order, int, position of the z_channel (off-plane axis)
            --> shape, np.array, three numbers (ints) representing shape of the volume
            --> cropped_shape, np.array, three numbers (ints) representing shape of the volume
            --> foreground, int, number representing the number of forground pixels
            --> fg_to_bg_ratio, float, ratio of pixels of foreground to the pixels of background
            --> class_statistics, dictionary, dictionary conatining number of voxels belonging to 
            each class
            --> voxel_statistics, dictionary, dictionary contatining statistics about voxels 
            (mean, std, percentiles, min, max)
            --> voxels_values, np.array, number_of_voxles long array containing random voxels 
            of the foreground
        """
        # Read Volume
        _input_volume = sitk.ReadImage(ct_data_path)
        _volume_numpy_array = sitk.GetArrayFromImage(_input_volume)    
        _volume_numpy_array = rearrange(_volume_numpy_array, 'z y x -> x y z')

        # Read mask
        _segmentation_mask = sitk.ReadImage(seg_mask_path)
        _segmentation_numpy_array = sitk.GetArrayFromImage(_segmentation_mask)    
        _segmentation_numpy_array = rearrange(_segmentation_numpy_array, 'z y x -> x y z')
        _segmentation_numpy_array = _segmentation_numpy_array.astype(np.int16)

        # Obtain number of channels
        _number_of_channels = _input_volume.GetNumberOfComponentsPerPixel()

        # Estimate ordering of the z channel
        _direction = _input_volume.GetDirection()
        _z_channel_order = 0
        _x_direction, _y_direction, _z_direction = _direction[0:3], _direction[3:6], _direction[6:9]

        if _x_direction == (0, 0, 1) or _x_direction == (0, 0, -1):
            _z_channel_order = 1
        if _y_direction == (0, 0, 1) or _y_direction == (0, 0, -1):
            _z_channel_order = 2
        if _z_direction == (0, 0, 1) or _z_direction == (0, 0, -1):
            _z_channel_order = 3

        # Obtain spacing
        _spacing = np.array(_input_volume.GetSpacing())

        # Crop to non-zero content (C, X, Y, Z) is required
        _cropped_volume, _cropped_segmentation_mask, _bbox = crop_to_nonzero(
            data = np.expand_dims(_volume_numpy_array, 0),
            seg = np.expand_dims(_segmentation_numpy_array, 0),
            nonzero_label = -1)

        # Squeeze added dim
        _cropped_volume = np.squeeze(_cropped_volume, 0)
        _cropped_segmentation_mask = np.squeeze(_cropped_segmentation_mask, 0)

        # Obtain data
        _shape = _volume_numpy_array.shape
        _cropped_shape = _cropped_volume.shape
        _fg_to_vol_ratio = np.count_nonzero(_cropped_segmentation_mask) / _volume_numpy_array.size

        # For labels population
        _foreground_per_class_statistics = Counter(_cropped_segmentation_mask.flatten())

        # Obtain Voxel Statistics only on foreground
        # Get non-zero indices from foreground
        _non_zero_indices = _cropped_segmentation_mask > 0
        # Obtain values
        _voxel_values = _cropped_volume[_non_zero_indices]

        # Obtain voxel statistics
        _mean, _mini, _maxi = np.nan, np.nan, np.nan
        _std, _percentile_00_5, _percentile_99_5, _median =  np.nan, np.nan, np.nan, np.nan
        _sampled_voxels = []

        # Obtain random number of voxels from the foreground
        _number_of_fg_voxels = len(_voxel_values)

        # Calculate values only if the image contatins fg voxels
        if _number_of_fg_voxels > 0:
            _percentile_00_5, _median, _percentile_99_5 = np.percentile(
                _voxel_values, 
                np.array([0.5, 50, 99.5]))
            _mean = np.mean(_voxel_values)
            _mini = np.min(_voxel_values)
            _maxi = np.max(_voxel_values) 
            _std = np.std(_voxel_values)     

            # Obtain foreggrounds pixels
            _rs = np.random.RandomState(1993)
            _sampled_voxels = list(_rs.choice(_voxel_values, number_of_voxels, replace=True))

        # Create dict
        _voxel_statistics_dict = {
            'mean': _mean,
            'median': _median,
            'min': _mini,
            'max': _maxi,
            'std': _std,
            'percentile_99_5': _percentile_99_5,
            'percentile_00_5': _percentile_00_5,
        }
        # Generate export dict
        _export_dict = {"spacing": _spacing,
                        "channels": _number_of_channels,
                        "z_channel_order": _z_channel_order,
                        "shape": _shape,
                        "cropped_shape" : _cropped_shape,
                        "fg_to_vol_ratio": _fg_to_vol_ratio,
                        "class_statistics": _foreground_per_class_statistics,
                        "voxel_statistics": _voxel_statistics_dict,
                        "voxel_values": _sampled_voxels}       
        # Return
        return _export_dict

    def __obtain_axis_sizes(
        self)->dict:
        """
        Method which obtains dataset properties such as sizes, medians, fg/bg ratios etc.

        Args:
            * None, it uses data stored in self.CT_files and self.seg_maps.
        
        Returns:
            * Dictionary (dict), contatining properties:
            --> Median_axis_size
            --> Mean_axis_size
            --> Median_axis_size_cropped
            --> Mean_axis_size_cropped
            --> Median_spacing_value
            --> Mean_spacing_value
            --> Median_fg_to_bg_ratio
            --> Mean_fg_to_vol_ratio
            --> Mean_number_of_channels (THIS SHOULD BE A WHOLE NUMBER, IF NOT CHECK DATA)
            --> Mean_z_channel_order (THIS SHOULD BE A WHOLE NUMBER, IF NOT CHECK DATA)

            * Dictionary (dict), containing voxel properties caluclated based on 10000 foreground
            samples randomly chosen from each CT. (10000 * num_CT)
            --> min
            --> max
            --> mean
            --> std
            --> 0.05 percentile
            --> 50.0 percetnile (medain)
            --> 99.5 percentile

            * Dictionary (dict), conataining voxel properties (same as dict before), but calculated 
            as a mean value of calculated values for each CT taking into account all foreground 
            voxels of observed CT (e.g. min = mean(list_of_all_min_values))

            * List of dictionary containing foreground count for each sample as well as it size
            This is to save time. It is not a dictionary with list of values for each key because 
            the information of the image size (total number of voxels) is lost
        """
        # Iterate through files
        _shapes_tmp = []
        _shapes_cropped_tmp = []
        _fg_vol_tmp = []
        _spacing_tmp = []
        _foreground_count_tmp = []
        _voxel_statistic_tmp = []
        _channels_tmp = []
        _z_channel_order_tmp = []
        _voxel_values_tmp = []

        # Single processing
        if self.number_of_processes is None:
            for _ct_file, _seg_file in tqdm(zip(self.CT_files, self.seg_maps),
                                            total = len(self.CT_files)):
                # Update storage
                _result_dict = self._analyise_one_case(
                    ct_data_path = _ct_file,
                    seg_mask_path = _seg_file,
                    number_of_voxels = self.number_of_fg_voxels_to_evaluate
                    )
                _spacing_tmp.append(_result_dict["spacing"])
                _channels_tmp.append(_result_dict["channels"])
                _z_channel_order_tmp.append(_result_dict["z_channel_order"])
                _shapes_tmp.append(_result_dict["shape"])
                _shapes_cropped_tmp.append(_result_dict["cropped_shape"])
                _fg_vol_tmp.append(_result_dict["fg_to_vol_ratio"])
                _foreground_count_tmp.append(_result_dict["class_statistics"])
                _voxel_statistic_tmp.append(_result_dict["voxel_statistics"])
                _voxel_values_tmp += _result_dict["voxel_values"]
        else:
            ## Multiprocesing
            # Create progress bar & storage
            _progress_bar = tqdm(total=len(self.CT_files))
            _result_list = []
            # Creat input data
            _all_input_data = [*zip(self.CT_files, 
                                    self.seg_maps, 
                                    [self.number_of_fg_voxels_to_evaluate]*len(self.seg_maps))]
            with multiprocessing.Pool(processes = self.number_of_processes) as _pool:
                # Submit tasks asynchronously and use the callback to update the progress
                for _input_data in _all_input_data:
                    _result = _pool.apply_async(
                        self._analyise_one_case, 
                        args=_input_data, 
                        callback=lambda _: _progress_bar.update()) ##pool.map()
                    _result_list.append(_result)

                # Close the pool to prevent any more tasks from being submitted
                _pool.close()

                # Wait for all worker processes to finish
                _pool.join()

            # Collect the results
            _result_list = [_r.get() for _r in _result_list]
            _progress_bar.close()

            ## Devide result_list into smaller lists
            for _result_dict in _result_list:
                _spacing_tmp.append(_result_dict["spacing"])
                _channels_tmp.append(_result_dict["channels"])
                _z_channel_order_tmp.append(_result_dict["z_channel_order"])
                _shapes_tmp.append(_result_dict["shape"])
                _shapes_cropped_tmp.append(_result_dict["cropped_shape"])
                _fg_vol_tmp.append(_result_dict["fg_to_vol_ratio"])
                _foreground_count_tmp.append(_result_dict["class_statistics"])
                _voxel_statistic_tmp.append(_result_dict["voxel_statistics"])
                _voxel_values_tmp += _result_dict["voxel_values"]

        # Create export dictionary for voxel statistics
        _voxel_values_tmp = np.array(_voxel_values_tmp)
        # Merge dictionaries
        _voxel_statistics_dict = {}
        for _single_case_statistics in _voxel_statistic_tmp:
            for _key, _value in _single_case_statistics.items():
                if _key in _voxel_statistics_dict:
                    _voxel_statistics_dict[_key].append(_value)
                else:
                    _voxel_statistics_dict[_key] = [_value]

        # Calculated general properties of the given CTs
        _export_dict_general_info = {
            "Median_axis_size": np.median(np.array(_shapes_tmp), 0).astype(float).tolist(),
            "Mean_axis_size": np.mean(np.array(_shapes_tmp), 0).astype(float).tolist(),
            "Median_axis_size_cropped": np.median(np.array(_shapes_cropped_tmp), 0).astype(float).tolist(),
            "Mean_axis_size_cropped": np.mean(np.array(_shapes_cropped_tmp), 0).astype(float).tolist(),
            "Median_spacing_value": np.median(np.array(_spacing_tmp), 0).astype(float).tolist(),
            "Mean_spacing_value": np.mean(np.array(_spacing_tmp), 0).astype(float).tolist(),
            "Median_fg_to_bg_ratio": np.median(np.array(_fg_vol_tmp)).astype(float).tolist(),
            "Mean_fg_to_bg_ratio": np.mean(np.array(_fg_vol_tmp).astype(float).tolist()),
            "Mean_number_of_channels": np.mean(np.array(_channels_tmp).astype(float).tolist()),
            "Mean_z_channel_order": np.mean(np.array(_z_channel_order_tmp).astype(float).tolist())
        }

        # Calculated based on the 10e7 Randomly selected samples from each CT case
        _percentile_00_5, _median, _percentile_99_5 = np.percentile(
            _voxel_values_tmp,
             np.array([0.5, 50, 99.5]))
        _export_dict_voxel_info_1 = {
            "min": float(np.min(_voxel_values_tmp)),
            "max": float(np.max(_voxel_values_tmp)),
            "mean": float(np.mean(_voxel_values_tmp)),
            "std": float(np.std(_voxel_values_tmp)),
            "percentile_00_5": float(_percentile_00_5),
            "median": float(_median),
            "percentile_99_5": float(_percentile_99_5)
        }

        # Calculated as mean of a propertiy(ies) in all CTs cases. For insnance:
        # min = mean(all_min_in_all_CT)
        _export_dict_voxel_info_2 = {
            "min": float(np.mean(_voxel_statistics_dict["min"])),
            "max": float(np.mean(_voxel_statistics_dict["max"])),
            "mean": float(np.mean(_voxel_statistics_dict["mean"])),
            "std": float(np.mean(_voxel_statistics_dict["std"])),
            "percentile_00_5": float(np.mean(_voxel_statistics_dict["percentile_00_5"])),
            "median": float(np.mean(_voxel_statistics_dict["median"])),
            "percentile_99_5": float(np.mean(_voxel_statistics_dict["percentile_99_5"]))
        }

        # Return:
        return (_export_dict_general_info, 
                _export_dict_voxel_info_1,
                _export_dict_voxel_info_2,
                _foreground_count_tmp)

    def __obtain_class_properties(self, foreground_statistics_list: list = None)->dict:
        """
        Method which obtains statistics per class present in the dataset. 

        Args:
            * foreground_statistics_list, list, list containing per sample statistics. 
            It is obtainable by self.__obtain_axis_sizes method. The reason for doing 
            so is because reading and transformim SITK.image takes a lot of time 
            (no need for reread of segmentation maps)

        Returns:
            * dict for each class where for each class the following properties are calculated
            --> Number_of_samples: int, No of samples in the provided set
            --> Median_size: float, median number of voxels bellonging to this class
            --> Mean_size: float, mean number of voxels bellonging to this class
            --> Median_class_to_vol_ratio: float (ratio), medina of class to volume in all samples
            --> Mean_class_to_vol_ratio: float (ratio), mean of class to volume  in all samples     
            --> Median_class_to_fg_ratio: float (ratio), median of class to foreground in all 
            samples
            --> Mean_class_to_fg_ratio: float (ratio), mean of class to foreground in all samples
        """
        # Storage
        _classes_count = {}
        _classes_sizes = {}
        _classes_class_to_vol_ratio = {}
        _classes_class_to_fg_ratio = {}

        # Go thorugh cases and build statistics
        for _case_report in foreground_statistics_list:
            # Obtain total number of voxels in given case
            _total_num_of_voxels = sum(_case_report.values())
            _foreground_num_of_voxels = _total_num_of_voxels - _case_report[0]

            # Skip if only background
            if _foreground_num_of_voxels == 0:
                continue

            for _key in _case_report:
                # Update count
                if _key in _classes_count:
                    _classes_count[_key] +=1
                else:
                    _classes_count[_key] = 1

                # Update sizes
                if _key in _classes_sizes:
                    _classes_sizes[_key].append(_case_report[_key])
                else:
                    _classes_sizes[_key] = [_case_report[_key]]

                # Update class to vol ratio
                if _key in _classes_class_to_vol_ratio:
                    _classes_class_to_vol_ratio[_key].append(
                        _case_report[_key]/_total_num_of_voxels)
                else:
                    _classes_class_to_vol_ratio[_key] = [_case_report[_key]/_total_num_of_voxels]

                # Update class to fg ratio
                if _key in _classes_class_to_fg_ratio:
                    _classes_class_to_fg_ratio[_key].append(
                        _case_report[_key]/_foreground_num_of_voxels)
                else:
                    _classes_class_to_fg_ratio[_key] = [
                        _case_report[_key]/_foreground_num_of_voxels]

        # Build dict report
        _main_export_dict = {}
        for _key in _classes_count:
            _export_dict_for_class = {
                "Number_of_samples": _classes_count[_key],
                "Median_size": float(np.median(_classes_sizes[_key])),
                "Mean_size": float(np.mean(_classes_sizes[_key])),
                "Median_class_to_vol_ratio": float(np.median(_classes_class_to_vol_ratio[_key])),
                "Mean_class_to_vol_ratio": float(np.mean(_classes_class_to_vol_ratio[_key])),
                "Median_class_to_fg_ratio": float(np.median(_classes_class_to_fg_ratio[_key])),
                "Mean_class_to_fg_ratio": float(np.mean(_classes_class_to_fg_ratio[_key]))
            }
            # Update
            _main_export_dict[_key] = _export_dict_for_class

        # Return
        return _main_export_dict

    def export_fingerprint(self, export_path: str = None):
        """
        Method for exporting the fingerprint of the dataset. Fingerprint consist of 
        following properties:
        
        General_properites:
            --> Number_of_samples: int, No of samples in provided set
            --> Foreground_mean: float, mean value of all foreground pixels
            --> Foreground_std: float, float value of all foreground pixels
            --> Foreground_005_percentile, float, 0.5 percentile of all foreground pixels
            --> Foreground_median_percentile, float 50.0 percentile of all foreground pixels
            --> Foreground_995_percentile, float, 99.5 percentile of all foreground pixels
            --> Median_axis_size: tupple(int, int, (int)) for each axis of the dataset
            --> Mean_axis_size: tupple(int, int, (int)) for each axis of the dataset
            --> Median_axis_size_cropped: tupple(int, int, (int)) for each axis of the dataset 
            after cropping
            --> Mean_axis_size_cropped: tupple(int, int, (int)) for each axis of the dataset 
            after cropping
            --> Median_resample_value: tupple(int, int, (int)) for each axis of the dataset
            --> Mean_resample_value: tupple(int, int, (int)) for each axis of the dataset
            --> Median_fg_to_bg_ratio: float (ratio), ratio of avg fg/bg in all samples
            --> Mean_fg_to_bg_ratio: float (ratio), ratio of avg fg/bg in all samples
            --> Mean_number_of_channels (THIS SHOULD BE A WHOLE NUMBER, IF NOT CHECK DATA)
            --> Mean_z_channel_order (THIS SHOULD BE A WHOLE NUMBER, IF NOT CHECK DATA)
        
        Labels: just dict of voxel_value--> label_name pairs
            
        Voxel statistcs: (on 10000*num_of_CT samples)
            --> min
            --> max
            --> mean
            --> std
            --> 0.05 percentile
            --> 50.0 percetnile (median)
            --> 99.5 percentile

        Voxel statistics: (as mean of individual CT sampls):
            --> min
            --> max
            --> mean
            --> std
            --> 0.05 percentile
            --> 50.0 percetnile (median)
            --> 99.5 percentile
        
        Class_properties: (For each class)
            --> Number_of_samples: int, No of samples in the provided set
            --> Median_size: float, median number of voxels bellonging to this class
            --> Mean_size: float, mean number of voxels bellonging to this class
            --> Median_class_to_vol_ratio: float (ratio), medina of class to volume in all samples
            --> Mean_class_to_vol_ratio: float (ratio), mean of class to volume  in all samples     
            --> Median_class_to_fg_ratio: float (ratio), median of class to foreground in all 
            samples
            --> Mean_class_to_fg_ratio: float (ratio), mean of class to foreground in all samples          
        """
        # Check if file is update
        assert export_path is not None, \
            f"Export path must not be None! Please provide export path!"

        ### Go through properties and fill up each
        _general_properties_dict = {}
        # Sample number
        _general_properties_dict["Number_of_samples"] = len(self.CT_files)

        # Axis sizes & foreground ratios
        _dict_of_sizes, _dict_voxel_stats_samples, \
        _dict_voxel_stats_means, _foreground_statistics_list = self.__obtain_axis_sizes()

        # Update
        _general_properties_dict = _general_properties_dict | _dict_of_sizes

        # Obtain Classes statistics
        _dict_class_statistics = self.__obtain_class_properties(
            foreground_statistics_list = _foreground_statistics_list)

        # Export
        if os.path.exists(export_path):
            print(f"Output_file {export_path} already exists. Overriding it!")

        with open(export_path, 'w') as _f:
            # Write each dictionary separately
            for _d, _info in zip(
                [_general_properties_dict,
                self.labels,
                _dict_voxel_stats_samples,
                _dict_voxel_stats_means,
                _dict_class_statistics],
                ["GENERAL_INFO",
                "LABELS_MAPPING", 
                "VOXEL_STATISTICS_SAMPLES", 
                "VOXEL_STATISTICS_MEANS", 
                "CLASS_STATISTICS"]):

                _f.write(f"{_info}\n")
                # Format writing so humans can read JSON file :)
                _f.write('{\n')
                for _i, (_key, _value) in enumerate(_d.items()):
                    # Convert value to JSON format (handling nested dictionaries and lists)
                    _json_value = json.dumps(_value)

                    # Write the key-value pair in the desired format
                    _f.write(f'  "{_key}": {_json_value}')

                    # Add a comma after each pair except the last one
                    if _i < len(_d) - 1:
                        _f.write(',\n')
                    else:
                        _f.write('\n')

                # Write the closing brace for the JSON object
                _f.write('}\n')
                _f.write('\n')
                _f.write('**********************************************************************')
                _f.write('\n')
