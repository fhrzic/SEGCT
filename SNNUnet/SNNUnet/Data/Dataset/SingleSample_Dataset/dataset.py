""" Single sample dataset implementation in order to build
Dataset which passes through one sample in non overlapping 
patches manner
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


class SingleSampleDataset(object):
    """
    Class for providing the dataset

    """
    def __init__(self, dataset_config: DatasetConfig = None):
        """
        Init function of SingleSample. It accepts as input dataset_config so please refere to 
        dataset_config.py script for all details about attributs in the class. Short: 
        all necessary to do the dataset

        Args:
            * dataset_config: DatasetConfig, all information necessary to build dataset
        """
        # Set seed
        random.seed(1993)

        # Obtain paths to ct, seg maps and locations
        _ct_path = dataset_config.ct_path
        _seg_path = dataset_config.seg_path
        assert (_ct_path is not None and _seg_path is not None), \
            f"Paths to directories are not valid, please check them in provided dataset_config"

        # Obtain lists
        _ct_file_list = [os.path.join(_ct_path, _f) for _f in os.listdir(_ct_path)]
        _seg_file_list = [os.path.join(_seg_path, _f) for _f in os.listdir(_seg_path)]

        # Sort them so we get ensured for corresponding files
        _ct_file_list.sort()
        _seg_file_list.sort()

        # Apply limits
        if dataset_config.sample_limit is None:
            self.ct_file_list = _ct_file_list
            self.seg_file_list = _seg_file_list
        else:
            _limit = min(dataset_config.sample_limit, len(_ct_file_list))
            self.ct_file_list = _ct_file_list[0:_limit]
            self.seg_file_list = _seg_file_list[0:_limit]

        # Obtain patch size
        self.patch_shape = dataset_config.patch_shape

        # Obtain multiprocessing
        self.multiprocessing = dataset_config.multiprocessing

        # Obtain augmentations
        self.augmentation = dataset_config.augmentation_method

        # Obtain self backgoround label
        self.background_label, self.ignore_label = self.__obtain_background_and_ignore_labels(
            path_to_fingerprint = dataset_config.data_fingerprint_path)

        # Obtain files if keep in memory is true
        self.keep_in_memory = dataset_config.keep_in_memory
        if self.keep_in_memory:
            # Storages
            self.ct_preloaded_dict = {}
            self.seg_maps_preloaded_dict = {}
            # Load
            self.__preload_volumes_and_segmentations()

    def __obtain_background_and_ignore_labels(self, path_to_fingerprint: str = None)->int:
        """
        Method for obtaining the background label value

        Args:
            * path_to_fignerprint, string, path to the fingerprint file

        Returns:
            * two floats: first on is background label, while the second one is ignore label one.
        """
        # Read data and obtrain json
        try:
            _dictionaries = read_data_fingerprint_json(path_to_fingerprint)
            _classes = _dictionaries["LABELS_MAPPING"]
            _background_label = [_key for _key, _value in _classes.items()
                                 if _value == "background"][0]
            _ignore_label = [_key for _key, _value in _classes.items()
                             if _value == "ignore_label"][0]
        except ValueError as _exc:
            raise ValueError("Something went wrong in background and igore value obtaining."
                             f"Could not obtain data from {path_to_fingerprint}!") from _exc
        # Return
        return float(_background_label), float(_ignore_label)

    def _worker_for_obtaining_volumes_and_segmentations(self, idx: int = None)->list:
        """
        For multiprocessing reasons, this is the method which enables it 
        (multiprocessing needs middleman).

        Args:
            * idx, int, index of a file in self.seg_file_list == path to the 
            segmentation map(volume)

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
        As the name of the method says, this script preloads into RAM all CTs and all SEG maps
        It also supports multiprocessing. 

        Returns:
            * Nothing, but now there is now everything preloaded into RAM. 
        
        """

        # Get indexes of the files 
        _idx_list = list(range(len(self.seg_file_list)))

        # Go through list of segmentation maps
        if self.multiprocessing is None:
            for _idx in tqdm(_idx_list):
                _ct, _seg = self._worker_for_obtaining_volumes_and_segmentations(idx = _idx)

                # Update storages
                self.ct_preloaded_dict[_idx] = _ct
                self.seg_maps_preloaded_dict[_idx] = _seg
        else:
            # Multiprocessing is on
            with multiprocessing.Pool(self.multiprocessing) as _pool:
                _results = _pool.map(
                    self._worker_for_obtaining_volumes_and_segmentations, 
                    _idx_list)

            # Update storages
            for _idx, _data in enumerate(_results):
                _ct_volume, _seg_mask  = _data

                # Convert to torch tensors
                _ct_volume = torch.from_numpy(_ct_volume)
                _ct_volume = _ct_volume.to(torch.float32)
                _seg_mask = torch.from_numpy(_seg_mask)
                _seg_mask = _seg_mask.long()

                # Pad if necessary
                _ct_patches, _, _ = self.__symetrical_padding(volume = _ct_volume,
                                                        desired_shape = self.patch_shape,
                                                        patch_shape = self.patch_shape,
                                                        divisible_by_patch = False,
                                                        padding_value = self.background_label)

                _seg_mask_patches, _, _ = self.__symetrical_padding(volume = _seg_mask,
                                                        desired_shape = self.patch_shape,
                                                        patch_shape = self.patch_shape,
                                                        divisible_by_patch = False,
                                                        padding_value = self.ignore_label)

                # Devide it into patches
                _ct_patches, _ = self.__split_volume_into_patches_no_padding(
                    volume = _ct_patches,
                    patch_shape = self.patch_shape)
                _seg_mask_patches, _ = self.__split_volume_into_patches_no_padding(
                    volume = _seg_mask_patches,
                    patch_shape = self.patch_shape)


                self.ct_preloaded_dict[_idx] = _ct_patches
                self.seg_maps_preloaded_dict[_idx] = _seg_mask_patches

    def __symetrical_padding(self,
                             volume: torch.tensor = None, 
                             desired_shape: tuple = None, 
                             patch_shape: tuple = None,
                             divisible_by_patch: bool = False,
                             padding_value: int = 0) -> Tuple[torch.tensor, tuple, tuple]:
        """
        Method which adds padding symetrically (symetrical as possible) before and after the volume 
        until it is fully devisible the desired_shape is reached. If the devisable_by_patch is True, 
        then it is padded more until it is devisable by patch.

        Args:
            * volume, torch.tensor, input volume which must be padded
            * desired_shape, tuple, shape which we want it to be. NOW, if the original volume shape 
            is in any dimension smaller than desired_shape, it is symetrically zero padded to it. 
            If it is bigger, it is left as it is.
            * patch_shape, tuple, shape of the patch wich is usefull if divisible_by_patch is 
            turned on.
            * divisiable_by_patch, bool, if set True, the volume is padded even more until in every 
            dimnesion it is divisable by corresponding dimension in the patch_shape
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
        if divisible_by_patch == True:
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

    def __split_volume_into_patches_no_padding(
        self,
        volume: torch.tensor = None,
        patch_shape: tuple = None,
        overlap: float = 0)->Tuple[torch.tensor, list]:
        """
        Split the 3D volume into (non-)overlapping patches of shape self.patch_shape. But instead 
        of zero padding the volume to the closes devisable shape, it does reverse. It creates all 
        normal patches, then it makes patches on the residuals. E.G. On 1D example let's say that
        volume is 130px long and patch is 100. It will make two patches frist one from 0-100, and 
        second from 30-130. It Will not zero pad the volume to 200 and then make two patches: from 
        0-100 and 100-200.

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

    def __getitem__(self, idx: int):
        """
        Get method which samples from the data. and returns one sample.
         
        Args:
            * idx, int, indx of the data which is going to be processed

        Returns:
            * Sample
        """
        # Load volumes and segs maps
        if self.keep_in_memory:
            _ct_patches, _seg_mask_patches = self.ct_preloaded_dict[idx], self.seg_maps_preloaded_dict[idx]
        else:
            _ct_volume, _seg_mask = self._worker_for_obtaining_volumes_and_segmentations(idx=idx)
            # Convert to torch tensors
            _ct_volume = torch.from_numpy(_ct_volume)
            _ct_volume = _ct_volume.to(torch.float32)
            _seg_mask = torch.from_numpy(_seg_mask)
            _seg_mask = _seg_mask.long()
            # Pad if necessary
            _ct_volume, _padding_start, _original_volume_size  = self.__symetrical_padding(
                volume = _ct_volume,
                desired_shape = self.patch_shape,
                patch_shape = self.patch_shape,
                divisible_by_patch = False,
                padding_value = self.background_label)

            _seg_mask, _padding_start2, _original_volume_size2 = self.__symetrical_padding(
                volume = _seg_mask,
                desired_shape = self.patch_shape,
                patch_shape = self.patch_shape,
                divisible_by_patch = False,
                padding_value = self.ignore_label)

            # Devide it into patches
            _ct_patches, _ = self.__split_volume_into_patches_no_padding(
                volume = _ct_volume,
                patch_shape = self.patch_shape)
            _seg_mask_patches, _ = self.__split_volume_into_patches_no_padding(
                volume = _seg_mask,
                patch_shape = self.patch_shape)

        # Return
        return _ct_patches, _seg_mask_patches

    def __len__(self):
        # Length of the dataset(num_of_cases)
        return len(self.ct_file_list)
