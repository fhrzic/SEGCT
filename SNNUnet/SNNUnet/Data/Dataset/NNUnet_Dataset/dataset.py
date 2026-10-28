"""  
Dataset implementation script
"""
import torch
import numpy as np
import random
import os
from tqdm import tqdm
import multiprocessing
import torch.nn.functional as F
from typing import Tuple, List
import sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json
from Configurations.config_data import DatasetConfig


#**********************************************************************#
# Main class for data loading
#**********************************************************************#

class PatchDataset(object):
    """
    Class for providing the dataset

    """
    def __init__(self, dataset_config: DatasetConfig = None):
        """
        Init function of PatchDataset. It accepts as input dataset_config so please refere to 
        dataset_config.py script for all details about attributs in the class. Short: all necessary 
        to do the dataset.

        Args:
            * dataset_config: DatasetConfig, all information necessary to build dataset
        """
        # Set seed
        random.seed(1993)

        # Obtain paths to ct, seg maps and locations
        _ct_path = dataset_config.ct_path
        _seg_path = dataset_config.seg_path
        _loc_path = dataset_config.loc_path
        assert (_ct_path != None and _seg_path != None and _loc_path != None), \
            f"Paths to directories are not valid, please check them in provided dataset_config"

        # Obtain lists
        _ct_file_list = [os.path.join(_ct_path, _f) for _f in os.listdir(_ct_path)]
        _seg_file_list = [os.path.join(_seg_path, _f) for _f in os.listdir(_seg_path)]
        _loc_file_list = [os.path.join(_loc_path, _f) for _f in os.listdir(_loc_path)]
        # TODO Filter loc files
        _loc_file_list = [_path for _path in _loc_file_list if "surfaces" not in _path]

        # Sort them so we get ensured for corresponding files
        _ct_file_list.sort()
        _seg_file_list.sort()
        _loc_file_list.sort()
        
        # Check for the whitelist
        if dataset_config.white_list_cases != None:
            _ct_file_list_new = []
            _seg_file_list_new = []
            _loc_file_list_new = []
            for _idx, (_ct, _seg, _loc) in enumerate(zip(_ct_file_list, _seg_file_list, _loc_file_list)):
                if os.path.basename(_ct).split(".")[0] in dataset_config.white_list_cases:
                    _ct_file_list_new.append(_ct)
                    _seg_file_list_new.append(_seg)
                    _loc_file_list_new.append(_loc)
            # Update
            _ct_file_list = _ct_file_list_new
            _seg_file_list = _seg_file_list_new
            _loc_file_list = _loc_file_list_new

        # Apply limits
        if dataset_config.sample_limit is None:
            self.ct_file_list = _ct_file_list
            self.seg_file_list = _seg_file_list
            self.loc_file_list = _loc_file_list
        else:
            _limit = min(dataset_config.sample_limit, len(_ct_file_list))
            self.ct_file_list = _ct_file_list[0:_limit]
            self.seg_file_list = _seg_file_list[0:_limit]
            self.loc_file_list = _loc_file_list[0:_limit]

        # Obtain patch size
        self.patch_shape = list(dataset_config.patch_shape)

        # Obtain minimum fg
        self.minimum_fg = dataset_config.minimum_fg

        # Obtain multiprocessing
        self.multiprocessing = dataset_config.multiprocessing

        # Obtain fingerprint dictionaries
        _fingerprint_dicts = read_data_fingerprint_json(
            path_to_json = dataset_config.data_fingerprint_path)

        # Obtain augmentations
        if dataset_config.augmentation_method is not None:
            assert dataset_config.augmentation_method.__class__.__name__ in ["NNUnetAugmentation"],\
                f"Augmentation method {dataset_config.augmentation_method} is not implemented!"
        
        # Obtain initial patch size
        if dataset_config.augmentation_method.__class__.__name__  == "NNUnetAugmentation":
            self.augmentation = dataset_config.augmentation_method
            self.initial_patch_shape = np.array(self.augmentation.get_initial_patch_size())
        else:
            self.initial_patch_shape = np.array(self.patch_shape)
            self.augmentation = None

        # Obtain labels
        self._labels = list(_fingerprint_dicts["LABELS_MAPPING"].keys())
        self.ignore_label = next((_k for _k, _v in _fingerprint_dicts["LABELS_MAPPING"].items()
                                  if _v == "ignore_label"), None)
        self.background_label = next((_k for _k, _v in _fingerprint_dicts["LABELS_MAPPING"].items()
                                      if _v == "background"), None)

        # Remove ignore label and background (potentially)
        self._labels.remove(self.ignore_label)
        if dataset_config.include_background_as_class is False:
            self._labels.remove(self.background_label)

        assert self.background_label != None or self.ignore_label != None,\
            f"ignore_label or background label is not specified in fingerprint dict. Please specify them!"

        ### For sampling we need two things:
        ## dictionary where key is class and value is list of indx which contain that class
        ## for each sample we need to obtain possible locations for each voxel with valid positions

        # Create dictionary which will have key = label, and value = [indexes of cases contating that label]
        self.cases_and_classes_dict = {}
        for _label in self._labels:
            self.cases_and_classes_dict[_label] = []

        # Preload everything what can be preloaded
        self.classes_per_case_dict = {}
        self.ct_preloaded_dict = {}
        self.seg_maps_preloaded_dict = {}
        self.opened_location_files = {}
        
        # Populate self.cases_and_clases_dict from above. 
        print("Loading locations")
        self.__populate_cases_and_classes()

        # Obtain files if keep in memory is true
        self.keep_in_memory = dataset_config.keep_in_memory
        if self.keep_in_memory:
            print("Loading volumes and segmentations")
            self.__preload_volumes_and_segmentations()          

    def _worker_for_obtaining_volumes_and_segmentations(self, idx: int = None)->list:
        """
        For multiprocessing reasons, this is the method which enables it 
        (multiprocessing needs middleman). There is overhead in obtaining case_storage_dict_tmp 
        (seg file is two times oppenend)... TODO

        Args:
            * idx, int, index of a file in self.seg_file_list == path to the segmentation 
            map(volume)

        Returns:
            * case_storage_dict_tmp, a dictionary where each key is a class and each value is a list
            of valid center voxel_patches
            * ct_array, np.ndarray, the array represnting original ct volume
            * segmentation array, np.ndarray, the array representing segmentation
        """

        # Load ct and segmentation arrays
        _ct_array = np.load(self.ct_file_list[idx])
        _seg_map_array = np.load(self.seg_file_list[idx])

        # Return
        return [_ct_array, _seg_map_array]

    def __preload_volumes_and_segmentations(self):
        """
        As the name of the method says, this script preloads into RAM all CTs and all SEG maps.
        It also supports multiprocessing. 

        Returns:
            * Nothing, but now there is now volumes and segmentations preloaded into RAM.  
        """
        # Obtain indexes which will be preloaded (based on their frequency or all)
        ## This is lambda function which sorts the indexes based on how many classes they contain
        _sorted_idxs = sorted(
            self.classes_per_case_dict.keys(),
            key=lambda _idx: len(self.classes_per_case_dict[_idx]), 
            reverse=True)
        _idx_to_be_loaded = int(self.keep_in_memory * len(_sorted_idxs))
        _idx_list = _sorted_idxs[0:_idx_to_be_loaded]

        # Go through list of segmentation maps
        if self.multiprocessing == None:
            for _idx in tqdm(_idx_list):
                _ct, _seg = self._worker_for_obtaining_volumes_and_segmentations(idx = _idx)
                # Update storages
                self.ct_preloaded_dict[_idx] = _ct
                self.seg_maps_preloaded_dict[_idx] = _seg
        else:
            # Multiprocessing is on
            # Obtain valid locations
            with multiprocessing.Pool(self.multiprocessing) as _pool:
                _results = _pool.map(
                    self._worker_for_obtaining_volumes_and_segmentations,
                    _idx_list)
            # Update storages
            for _data, _idx in zip(_results, _idx_list):
                _ct, _seg = _data
                self.ct_preloaded_dict[_idx] = _ct
                self.seg_maps_preloaded_dict[_idx] = _seg


    def _worker_for_obtaining_classes_in_case(self,
                                        idx: int = None)->List[str]:
        """
        For multiprocessing reasons, this is the method which enables it

        Args:
            * idx, int, index of a file in self.loc_file_list == path to 

        Returns:
            * list of keys presented in case
        """

        # Classes present in the case
        _case_present_classes = []

        # Load seg map
        _location_map_dict = np.load(self.loc_file_list[idx], mmap_mode = "r")
        # Get keys
        _case_present_classes = list(_location_map_dict.keys())

        # Return
        return _case_present_classes

    def __populate_cases_and_classes(self):
        """
        Method which obtain labels for each case 

        Args:
            NONE

        Returns:
            * it populates self.cases_and_clases_dict 
        """

        # Get indexes of the files 
        _idx_list = list(range(len(self.seg_file_list)))
        # Go through list of segmentation maps
        if self.multiprocessing == None:
            for _idx in tqdm(_idx_list):
                _case_present_classes = self._worker_for_obtaining_classes_in_case(idx = _idx)
                # Update storages
                for _key in self._labels:
                    if _key in _case_present_classes:
                        # key is label, value is list of cases with that label
                        self.cases_and_classes_dict[_key].append(_idx)
                        # key is case, value is list of labels in that case
                        self.classes_per_case_dict[_idx] = [_class for _class in _case_present_classes if _class != self.ignore_label]
        else:
            # Multiprocessing is on
            with multiprocessing.Pool(self.multiprocessing) as _pool:
                # Use tqdm to wrap the imap call for progress monitoring
                _results = []
                for result in tqdm(_pool.imap(self._worker_for_obtaining_classes_in_case, _idx_list), 
                                   total=len(_idx_list), 
                                   desc="Processing"):
                    _results.append(result)
            # Update storages
            for _idx, _case_present_classes in enumerate(_results):
                for _key in self._labels:
                    if _key in _case_present_classes:
                        # key is label, value is list of cases with that label
                        self.cases_and_classes_dict[_key].append(_idx)
                        # key is case, value is list of labels in that case
                        self.classes_per_case_dict[_idx] = [_class for _class in _case_present_classes if _class != self.ignore_label]

    def __get_patch(
        self,
        volume: np.ndarray = None, 
        segmentation: np.ndarray = None,
        lower_bounding_box: list = None,
        upper_bounding_box: list = None)->np.ndarray:
        """
        Method for cropping patch out of the volume.

        Args:
            * volume, np.ndarray, volume from which the patch will be cropped
            * segmentation, np.ndarray, representing segmentation
            * lower_bounding_box, list, list contation location of lower bounding box
            * upper_bounding_box, list, list contation location of upper bounding box
        
        Returns:
            * np.ndarray, patch on the given location. If there can not be created the valid 
            patch 
            * np.ndarray, segmentation mask on the given location. If there can not be created 
            the valid patch
        """
        # Obtain volum shape
        _volume_shape = volume.shape
        
        # Idea is to get first the the valid positions of the volume, slice it, and then pad the rest 
        _valid_bbox_lbs = np.clip(lower_bounding_box, a_min=0, a_max=None)
        _valid_bbox_ubs = np.minimum(_volume_shape, upper_bounding_box)

        # Expand them through first dimension
        volume = np.expand_dims(volume, axis=0)         
        segmentation = np.expand_dims(segmentation, axis=0) 

        # Get slices and slice data
        _slice = tuple([slice(0, volume.shape[0])] + [slice(_i, _j) for _i, _j in zip(_valid_bbox_lbs, _valid_bbox_ubs)])
        _volume_patch = volume[_slice]

        # Do the same for segmentation
        _slice = tuple([slice(0, segmentation.shape[0])] + [slice(_i, _j) for _i, _j in zip(_valid_bbox_lbs, _valid_bbox_ubs)])
        _segmentation_patch = segmentation[_slice]
        

        # Calculate padding
        _padding = [(-min(0, _valid_bbox_lbs[_i]), max(upper_bounding_box[_i] - _volume_shape[_i], 0)) for _i, _ in enumerate(_volume_shape)]
        _padding = ((0, 0), *_padding)
        _volume_patch = np.pad(_volume_patch, _padding, 'constant', constant_values= int(self.background_label))
        _segmentation_patch = np.pad(_segmentation_patch, _padding, 'constant', constant_values = int(self.ignore_label))

        # Remove dimension
        _volume_patch = np.squeeze(_volume_patch, axis=0)
        _segmentation_patch = np.squeeze(_segmentation_patch, axis=0)

        # Return
        return _volume_patch, _segmentation_patch

    def __obtain_patch_coordinates(
        self,
        idx: int = None,
        selected_class: str = None,
        volume_shape: List[int] = None)->Tuple[List[int], List[int]]:
        """
        Method which obtains random voxel based on provided index (which define case)
        and selects random patch from the volume. Also it performs necessary padding.

        Args:
            * index, int, index of the volume for which the sample is going to be selected
            * selected_class, str, if provided, then this class is selected from the volume 
            (patch for sure contains this class)
            * volume_shape, Tuple of ints, dimensions of the volume shape.

        Returns:
            * two lists: first list is the coordinate of the selected upper left corner, 
            second is a list of bottom right corner
        """
        # Caluclate the difference between wanted shape required by augumentation
        # and expected output shape by the model. This is the amount we have to pad
        _patch_shape_difference = self.initial_patch_shape - self.patch_shape
        # There can be a problem when required patch shape is bigger than the whole volume
        # E.G. required patch_shape is 200, 200, 250, and volume is 100, 100, 300, for 
        # augmentation we have calculate that we need 400, 400, 400 (difference is 200, 200 , 150)
        # So 100 + 200, 100 + 200, 150 + 300 is 300, 300, 450, we still missing 100 and 100 in first
        # two dimensions  We then need to pad even more.
        for _dim, _ in enumerate(volume_shape):  
            if volume_shape[_dim] + _patch_shape_difference[_dim] < self.initial_patch_shape[_dim]:
                _patch_shape_difference[_dim] = self.initial_patch_shape[_dim] - volume_shape[_dim]

        # Now we know how much we need to pad. Padding is symetrical, so we need to calculate 
        # padding intervals (low padding and high padding boundaries) for each dimension for worst 
        # case scenarios. If we chose the worst case, we will still get a valid patch
        _lpb = [-_patch_shape_difference[_i] // 2 for _i, _ in enumerate(volume_shape)]
        _upb = [volume_shape[_i] + _patch_shape_difference[_i] // 2 + _patch_shape_difference[_i] % 2 \
            - self.initial_patch_shape[_i] for _i, _ in enumerate(volume_shape)]

        # Random location
        if selected_class == "-1":
            # Now we need to select boundary box inside the caluclated padding intervals
            _b_box_lower_bound = [np.random.randint(_lpb[_i], _upb[_i]+1) for _i, _ in enumerate(volume_shape)]
            _b_box_upper_bound = [_b_box_lower_bound[_i] + self.initial_patch_shape[_i] for _i, _ in enumerate(volume_shape)]
        else:
            # We must select the class 
            # First we load for given index given class array
            if idx in self.opened_location_files:
                _all_fg_locations = self.opened_location_files[idx]
            else:
                self.opened_location_files[idx] = np.load(self.loc_file_list[idx], mmap_mode = "r")
                _all_fg_locations = self.opened_location_files[idx]

            _class_candidates_locations = _all_fg_locations[selected_class]

            # Get random location of a voxel which belongs to the selected class. 
            # Shape of _class_candidates_locations is (n, casses, 1, 3(locations))
            _random_voxel_location = np.random.choice(_class_candidates_locations.shape[0])
            _random_voxel_location = _class_candidates_locations[_random_voxel_location][0]

            # Now we caluclate lower and upper bound from the "perspective" of the voxel
            _b_box_lower_bound = [max(_lpb[_i], _random_voxel_location[_i] - self.initial_patch_shape[_i] // 2) \
                for _i, _ in enumerate(volume_shape)]
            _b_box_upper_bound = [_b_box_lower_bound[_i] + self.initial_patch_shape[_i] for _i, _ in enumerate(volume_shape)]

        # Return values
        return [_b_box_lower_bound, _b_box_upper_bound]
            
    def __load_sample(
        self,
        volume_path: str = None, 
        seg_mask_path: str = None)->list:
        """
        Method for lading the sample. Pay attention to mmap_mode - this flag saves about 6 min 
        on 7 min epoch and it is in collsion with any type of disccaching (2 weeks of my life)
        Args:
            * volume_path, str, path to the volume that should be loaded
            * seg_mask_path, str, path to the segmentation mask that should be loaded

        Returns:
            * list of two elements where first represent loaded volume path while second 
            represent loaded segmentation mask
        """

        # Load data into memory
        _volume_npy = np.load(volume_path, mmap_mode="r")
        _seg_mask_npy = np.load(seg_mask_path, mmap_mode="r")
        # Return
        return [_volume_npy, _seg_mask_npy]
    
    def __getitem__(self, input:dict):
        """
        Get method which samples from the data. and returns one sample.
         
        Args:
            * It accepts small dict which has two keys "index": int and "class":"int". If class is -1
            the we seek for the random location.

        Returns:
            * Sample
        """
        # Obtain volume for given index
        if self.keep_in_memory and input["index"] in self.ct_preloaded_dict.keys():
            _ct_volume, _seg_mask = self.ct_preloaded_dict[input["index"]], self.seg_maps_preloaded_dict[input["index"]]
        else:
            _ct_volume, _seg_mask = self.__load_sample(
                volume_path = self.ct_file_list[input["index"]], 
                seg_mask_path = self.seg_file_list[input["index"]])

        # Obtain random patch for that case and that class
        _lower_b_box, _upper_b_box = self.__obtain_patch_coordinates(idx = input["index"],
                                                            selected_class = input["class"],
                                                            volume_shape = np.array(_ct_volume.shape))
        # Obtain patches
        _ct_patch, _seg_mask_patch = self.__get_patch(volume = _ct_volume, 
                                                        segmentation = _seg_mask, 
                                                        lower_bounding_box = _lower_b_box,
                                                        upper_bounding_box = _upper_b_box)

        # Convert to torch tensors
        _ct_patch = torch.from_numpy(_ct_patch)
        _ct_patch = _ct_patch.to(torch.float32)
        _seg_mask_patch = torch.from_numpy(_seg_mask_patch)
        _seg_mask_patch = _seg_mask_patch.long()
        # Add channel
        _ct_patch = torch.unsqueeze(_ct_patch, 0)
        _seg_mask_patch = torch.unsqueeze(_seg_mask_patch, 0)

        # Apply augmentation (make also placeholder for ds)
        _ds_mask_patches = []
        if self.augmentation is not None:
            # Check for augmentation type.
            if self.augmentation.__class__.__name__ == "NNUnetAugmentation":
                # Apply aug
                _ct_patch, _seg_mask_patch, _ds_mask_patches = self.augmentation(
                    _ct_patch,
                    _seg_mask_patch)
            
                # Transfer mask_patches in deep suprevision to the appropriate size
                for _idx, _ in enumerate(_ds_mask_patches):
                    _ds_mask_patches[_idx] = torch.squeeze(_ds_mask_patches[_idx], 0)
                    _ds_mask_patches[_idx] = _ds_mask_patches[_idx].long()

        # Remove channel from mask (not necessary for loss) and cast it to int8(-127 to 127)

        # (some augumentation may change data type)
        _seg_mask_patch = torch.squeeze(_seg_mask_patch, 0)
        _seg_mask_patch = _seg_mask_patch.long()

        # Obtain indicators
        _idx = torch.tensor(input["index"])
        _stacked_array = np.stack([_lower_b_box, _upper_b_box])
        _patch_location = torch.tensor(_stacked_array)

        # Return
        return [_ct_patch, _seg_mask_patch,
                _ds_mask_patches, _idx,
                _patch_location, input["index"],
                input["class"]]

    def __len__(self):
        """
        This returns number of total samples in the dataset
        """
        return len(self.ct_file_list)

    def get_cases_and_classes(self)->dict:
        """
        Get method for labels
        """
        return self.classes_per_case_dict
