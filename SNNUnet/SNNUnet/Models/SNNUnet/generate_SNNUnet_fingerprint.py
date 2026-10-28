""" Generate fingerprint for the SNNUnet"""
import os
import sys
import json
import re
from typing import Tuple, List
import numpy as np
import torchinfo
import torch

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

# Imports which can be changed
from Models.SNNUnet.SNNUnet import SNNUnet
from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json
from Models.SNNUnet.autograd_and_gradient_clipping import DummyContext

class GenerateSNNUnetFingerprint():
    """
    Class for generating the SSNUnet fingerprint which has following attributes:
    
    1.) Input Patch Shape
    2.) Max Batch Size
    3.) Output shapes
    4.) Output spacing
    5.) Z_last_channel (as a bool)
    6.) Number of channels (as an int)
    7.) Number_of_output_classes
    8.) Input spacing

    This attributes are necessary for building the 
    """

    def __init__(
        self,
        dataset_fingerprint_path: str = None,
        device: str = None,
        device_limit: int = None,
        mixed_precision: bool = False):
        """
        Init function which accepts only four arguments necessary for creating optimal 
        model configuration as it was described in NNUnet/ First we find the max patch 
        size and then we estimate batch size.

        Args:
            * dataset_fingerprint_path, str, path to the dataset fingerprint generated 
            by the script GenerateDatasetFingerprint.py
            * device, str, name of the device where the model will be trained.
            * device_limit, int, if provided this is the limit in MB to what extent of the
            GPU we want for model to be used. For instance we have 16GB of ram GPU but we
            want only use 8GB for the model. This also applies to the max_batch estimation.
            Literally, the GPU is shrinken. 
            * mixed_precision, bool, if set to true the mixed precision will be used during
            max patch estimation for the model. This parameter needs to be propagated to the
            training also because it enables biger patch size and therefore bigger 
            memory consuption with the cost of lower precision of model weights.
        """
        # Check if file exists
        assert os.path.exists(dataset_fingerprint_path), \
            f"Wrong path to the dataset fingerprint. Please check path {dataset_fingerprint_path}!"

        # Obtain data from the dataset fingerprint
        _dataset_fingerprint = read_data_fingerprint_json(dataset_fingerprint_path)
        _median_size = _dataset_fingerprint["GENERAL_INFO"]["Median_axis_size"]
        self.median_spacing = _dataset_fingerprint["GENERAL_INFO"]["Median_spacing_value"]
        self.number_of_samples = _dataset_fingerprint["GENERAL_INFO"]["Number_of_samples"]
        _z_channel = _dataset_fingerprint["GENERAL_INFO"]["Mean_z_channel_order"]
        _number_of_channels = _dataset_fingerprint["GENERAL_INFO"]["Mean_number_of_channels"]
        # Due to the ignore label we go with -1
        self.number_of_output_classes = len(_dataset_fingerprint["LABELS_MAPPING"]) - 1

        # Check if z_channel and channel numbers are valid
        assert _z_channel.is_integer(), \
            "Mean number of Z channels is not integer (it has decimal places)." \
            + "This means that all CTs are not in the same orientation." \
            + "Please check it out in datasetfingerprint and preprocessing." \
            + f"Mean_z_channel_order = {_z_channel}"

        assert _number_of_channels.is_integer(), \
            "Mean number of channels is not integer (it has decimal places)." \
            + "This means that all CTs are not in 3 Channel or 1 Channeled" \
            + "Please check it out in datasetfingerprint and preprocessing." \
            + f"Mean_number_of_channels = {_number_of_channels}"
        # Now obtain correct z_channel_order and number of channels in general        
        assert int(_z_channel) != 2, \
            "Z channel must be first or last but got 2 (in the middle). This is not supported!"
        #_median_size = [ 816.5, 537.5, 539.0]
        #_z_channel = 1
        # Set number of channels
        if int(_z_channel) == 1:
            self.z_last_channel = False
        else:
            # It must be three so it is last
            self.z_last_channel = True

        # Channels
        self.number_of_channels = int(_number_of_channels)

        # Median spacing must be set to integers
        self.median_size = [int(_i) for _i in _median_size]

        # Set device
        self.device = torch.device(device)

        # Set device limit
        self.device_limit = device_limit

        # Set mixed precision
        self.mixed_precision = mixed_precision

        # Obtain candidate shapes
        _candidate_shapes_list = self.__obtain_candidate_shapes(
            median_shape = self.median_size,
            median_spacing = self.median_spacing)

        # Now we seek for the best fitting index
        _fitting_idx = self.__find_first_combination_that_fits(
            shapes_list = _candidate_shapes_list,
            input_spacing = self.median_spacing,
            number_of_channels = self.number_of_channels,
            z_last_channel = self.z_last_channel,
            number_of_output_classes = self.number_of_output_classes,
            device = device)

        # Finally obtain input shape
        self.input_patch_shape = _candidate_shapes_list[_fitting_idx]
        self.input_patch_shape = [int(_i) for _i in self.input_patch_shape]

    def __obtain_candidate_shapes(
        self,
        median_shape: Tuple[int,int,int] = None,
        median_spacing: Tuple[float, float, float] = None)->List[List[int]]:
        """
        Method which obtains candidate list of the patches. It is a simple list which estimates 
        patches for the SNNUnet. Estiamtion is done as close as to the NNUnet (I got the same 
        outputs for all tested dataset if I assumed correct device capacity in RAM). How it is 
        done... well I tried to explain it as close I could, but it is not trivial. 
        More comments to be followed. Oh yeah, we try to find the bigest one of the BATCH size 
        2 which fits. Batch size is then caluclated and adjusted.

        Args:
            * median_shape, Tuple[int, int, int], input shape of the data extracted from the 
            dataset fingerprint
            * median_shape, Tuple[float, float, float], input median shape extracted from the 
            dataset fingerprint

        Returns:
            * list, list of the possible candidates as List[List[int]] where each inner list 
            has three ints representing the size of input pathces.
        """

        # First part is to find the initial input size. Here I don't care about limitations set in
        # NNUnet. This implementation trully tries to find the bigest shape and put it on the GPU.
        # Get the inverse spacing and get the largest dimension
        _tmp = 1 / np.array(median_spacing)
        _max_dim = max(median_shape)

        # Calculate the initial shape
        initial_patch_shape = [round(_i) for _i in
                               _tmp * (_max_dim ** 3 / np.prod(_tmp)) ** (1 / 3)]

        # Clamp it so it does not exceed the initial median_shape
        _patch_shape = np.array([min(_i, _j) for _i, _j in
                                 zip(initial_patch_shape, median_shape[:len(median_spacing)])])

        # Perform halving nn-unet-way
        _number_of_halvings = self.__perform_halving_nnunet_way(input_patch_size = _patch_shape)

        # Estimate new _patch_shape which is a CANDIDATE for being the input dim
        _patch_shape = [self.__next_divisible(x = _x, y = _y) for _x, _y in
                        zip(_patch_shape, _number_of_halvings)]

        # Now lets find all CANDIDATES
        # Create candidate shapes
        _candidate_shapes_list = []
        # Seek for the candidates
        while True:
            _candidate_shapes_list.append(_patch_shape)

            # Obtain halving
            _number_of_halvings = self.__perform_halving_nnunet_way(input_patch_size = _patch_shape)

            # Find next divisable
            _patch_shape = [self.__next_divisible(x = _x, y = _y) for _x, _y in
                            zip(_patch_shape, _number_of_halvings)]

            # Find axis to be halved (biggest one in relation to inital shape)
            _axis_to_be_reduced = np.argsort([_i / _j for _i, _j in
                                              zip(_patch_shape,
                                                   median_shape[:len(median_spacing)])])[-1]

            # Obtain reduction
            _reduction = 2** _number_of_halvings[_axis_to_be_reduced]
            _patch_shape[_axis_to_be_reduced] -= _reduction

            # Breaking point if it become too small
            if _patch_shape[0] < 8 and _patch_shape[1] < 8 and _patch_shape[2] < 8:
                break

        # Return
        return _candidate_shapes_list

    def __perform_halving_nnunet_way(
        self,
        input_patch_size: Tuple[int,int,int] = None)->Tuple[int,int,int]:
        """
        Method which obtains number of halvings for the provided input_patch_size.

        Args:
            * input_patch_size, Tuple(int, int, int), input patch size
        
        Returns:
            * it returns number of halvings per axis (Tuple(int, int, int)) for
            each axis.
        """
        # Make copy
        _working_patch_shape = input_patch_size.copy()
        # Create storage
        _number_of_halvings = [0,0,0]
        while True:
            # Simulate halving
            _succesfull_halvings = 0
            for _i, _ in enumerate(_working_patch_shape):
                # Simulate halving
                _tmp_dimension = np.ceil(_working_patch_shape[_i] / 2)

                # Check if halving is permitted
                if _tmp_dimension > 4:
                    # Halving is permitted
                    _number_of_halvings[_i] += 1
                    _working_patch_shape[_i] = _tmp_dimension
                    _succesfull_halvings += 1

            # Stop condition
            if _succesfull_halvings == 0:
                break
        # Return
        return _number_of_halvings

    def __next_divisible(
        self,
        x : int = 0,
        y : int = 0)->int:
        """
        Method which calculates next divisible of x by th 2**y

        Args:
            * x, int, number for which we are seeking divisible
            * y, int, power of the 2 for which we are seeking divisible

        Returns:
            * integer (number) which represents first greater or equal divisible of the given number
        """
        # Check if x is already divisible by y
        if x % 2**y == 0:
            return x
        else:
            # Calculate the next number greater than x that is divisible by y
            return x + (2**y - (x % 2**y))

    def __find_first_combination_that_fits(
        self,
        shapes_list: list = None,
        input_spacing: Tuple[int,int,int] = None,
        number_of_channels: int = 1,
        number_of_output_classes: int = 7,
        z_last_channel = False,
        device: str = "cpu" )->Tuple[int,int,int]:
        """
        Method which finds first combination of shapes which fits on the GPU. It do it based on the
        image size.

        Args:
            * shapes_list, tuple[int,int,int], input shapes of the model which can be considered as 
            the candidates.
            * input_spacing, tuple[int,int,int], input spacing of the model
            * number_of_channels, int, number of channels in input image
            * number_of_output_classes, int, number of output classes
            * z_last_channel, bool, if z is last or first channel (necessary for some builds)
            * device, string, targeted device where the model will be trained

        Returns:
            * tuple of three integers representing the larger shape wich fits on the gpu with batch 
            size 2
        """
        # Binary search
        _left, _right = 0, len(shapes_list) - 1
        # Result in case that we do not find any candidate that fits
        _result = -1  # Default if no fitting element is found

        # Perform binary search with testing if it fits
        while _left <= _right:
            _mid = (_left + _right) // 2
            if self.__estimate_model_size(input_shape = shapes_list[_mid],
                                          input_spacing = input_spacing,
                                          batch_size = 2,
                                          number_of_channels = number_of_channels,
                                          z_last_channel = z_last_channel,
                                          number_of_output_classes = number_of_output_classes,
                                          device = device):
                _result = _mid  # Possible first fitting index
                _right = _mid - 1  # Search left half to find first occurrence
            else:
                _left = _mid + 1  # Search right half

        # Return
        return _result

    def __estimate_model_size(self, 
                              input_shape: Tuple[int,int,int] = None,
                              input_spacing: Tuple[int,int,int] = None,
                              batch_size: int = 2,
                              number_of_channels: int = 1,
                              z_last_channel: bool = False,
                              number_of_output_classes: int = 7,
                              device: str = "cpu")->bool:
        """
        Method which estimates size of the model(forward and backward pass). Auxiliary 
        predictions are set to true because they generate negligable amount of MB to the model(but they still do).

        Args:
            * input_shape, tuple[int,int,int], input shape of the model
            * input_spacing, tuple[int,int,int], input spacing of the model
            * batch_size, int, input batch size
            * number_of_channels, int, number of channels in input image
            * z_last_channel, bool, if z is last or first channel (necessary for some builds)
            * number_of_output_classes, int, number of output classes (output channels)
            * device, string, targeted device where the model will be trained

        Returns:
            * bool if model fits on targeted device
        """
        # Create dummy model
        _dummy_model = SNNUnet(input_shape = input_shape, 
                               input_spacing = input_spacing, 
                               number_of_channels = number_of_channels,
                               z_last_channel = z_last_channel,
                               number_of_output_classes = number_of_output_classes,
                               output_auxiliary_predictions = True)

        # Obtain model size
        try:
            with torch.autocast(device_type= self.device.type) \
                if self.device.type == 'cuda' and self.mixed_precision else DummyContext():
                _survey = torchinfo.summary(
                    model = _dummy_model,
                    input_size = (batch_size, number_of_channels, *input_shape),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    device = device,
                    verbose = False)

            _survey = str(_survey)
            _dummy_model = _dummy_model.cpu()
            del _dummy_model
            torch.cuda.empty_cache()
        except RuntimeError:
            # Cuda error and did not fit
            return False
        
        # Compare memory sizes more rigorious approach
        _model_size = self.__extract_estimated_size(_survey)
        # Set device size (if limit applies apply limit, otherwise calculate full GPU size)
        if self.device_limit is not None:
            _device_size = self.device_limit
        else:
            _device_size = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        
        # See of it fits
        if _model_size > _device_size:
            return False
        else:
            return True

    def __extract_estimated_size(
        self,
        survey:str = None)->float:
        """
        Method which extracts size of th model in MB

        Args:
            * survey, string, survey obtained by the torchinfo.summary

        Returns:
            * float, number representing size of the model
        """
        # Use regex to find the "Estimated Total Size (MB):" line and capture the numerical value
        _match = re.search(r"Estimated Total Size \(MB\): ([\d.]+)", survey)
        # If a match is found, return the extracted value as a float; otherwise, return None
        if _match:
            return float(_match.group(1))
        return None

    def __estimate_max_batch_size(self)->int:
        """
        Method which estimates biggest batch size which can be set on the given device.
        Also it put constraintes that batch size can not exceed 5% of total voxels in all training
        cases.

        So there are two constraints:
            1.) must fit in the device
            2.) must not exceed 5% all pixels in the training set. (This is for small datasets)

        Args:
            * None
        
        Returns:
            * int which represents maximum number of samples in a batch which satisfies both 
            constraints
        """
        # Set device limit to None
        _tmp_device_limit = self.device_limit
        self.device_limit = None

        # Calculate what is 5 percent of total voxels in training set
        _5_percent_voxels = self.median_size[0] * self.median_size[1] * self.median_size[2] \
                            * self.number_of_samples * 5 / 100

        # Find batch size
        _batch_size = 2
        _fits_on_gpu = True
        _number_of_voxels = self.input_patch_shape[0] * self.input_patch_shape[1] \
                            * self.input_patch_shape[2] * _batch_size

        # Find biggest batch size
        while (_number_of_voxels < _5_percent_voxels and _fits_on_gpu):
            # Increase batch size by 1
            _batch_size += 1

            # Get estimation on the gpu
            _fits_on_gpu = self.__estimate_model_size(input_shape = self.input_patch_shape, 
                                                      input_spacing = self.median_spacing, 
                                                      batch_size = _batch_size,
                                                      number_of_channels = self.number_of_channels,
                                                      z_last_channel = self.z_last_channel,
                                                      device = self.device)
            # Recalculate number of voxles
            _number_of_voxels = self.input_patch_shape[0] * self.input_patch_shape[1] \
                                * self.input_patch_shape[2] * _batch_size

        # Return device limit
        self.device_limit = _tmp_device_limit

        # Return batch size (-1 because last one did not fit)
        return _batch_size - 1

    def __obtain_output_shapes_and_spacings(
        self)->List[List[Tuple[int, int, int]]]:
        """
        Method which obtains output shapes and spacings from the model.

        Args:
            * nothing

        Returns:
            * two lists where first list represents output shapes and the last list represent output
            spaces (in order from the last outputed to the first outputed)
        """
        # Create dummy model
        _dummy_model = SNNUnet(input_shape = self.input_patch_shape, 
                               input_spacing = self.median_spacing, 
                               number_of_channels = self.number_of_channels,
                               z_last_channel = self.z_last_channel,
                               number_of_output_classes = self.number_of_output_classes)

        # Obtain shapes and spacing
        _shapes, _spacings = _dummy_model.get_output_shapes()

        # Delete model
        _dummy_model = _dummy_model.cpu()
        del _dummy_model
        torch.cuda.empty_cache()

        # Return shapes and spacings
        return [_shapes, _spacings]

    def obtain_fingerprint(
        self,
        export_path: str = None)->dict:
        """
        Method which returns fingerprint of the model:
            1.) Input Patch Shape
            2.) Max Batch Size
            3.) Output shapes
            4.) Output spacings
            5.) Z_last_channgel (as a bool)
            6.) Number of channels (as an int)
            7.) Number_of_output_classes
            8.) Input spacing
          
        Args:
            * export_path, str, if not None, it will save "model_fingerprint.json" to the target 
            location. "SNNUnet_fingerprint.json" as the name is added only if the path already 
            does not end with ".json". In that case we assume that the exact file is provided
        
        Returns:
            * dict, dictionary having the aforementioned attributes necessary to rebuild model 
            and use deep_supervision loss. The keys are as follows:
                "input_patch_shape"->tuple, which representes max patch that can be used on the gpu
                "max_batch_size"->int, max batch that can be stored on the gpu with constraint that 
                number of voxels in a batch must not exceed 5% of total number of voxels in all 
                training cases.
                "output_shapes"->list, list containging output shapes as (int, int, int) from last 
                output to the first one (bottleneck)
                "output_spacings"->list, list of output spacings (float,float,float) in same order
                 as output shapes. IDK why we need this but why not --> maybe for the papers...
                "z_last_channel"->bool, if the z is last channel (True), or first channel (False). 
                Z is offplane plane.
                "number_of_input_channels"->int, number of input channels
                "number_of_output_classes"->int, number of output channels (classes)
                "input_spacing"->tuple, tuple of floats which representes input spacings
        """
        # Estimate max batch for the given shape
        _max_batch_size = self.__estimate_max_batch_size()
        # Output_shapes and output spacings
        _output_shapes, _output_spacings = self.__obtain_output_shapes_and_spacings()

        # Build dictionary
        _output_dict = {
            "input_patch_shape" : self.input_patch_shape,
            "max_batch_size" : _max_batch_size,
            "output_shapes" : _output_shapes,
            "output_spacings" : _output_spacings,
            "z_last_channel" : self.z_last_channel,
            "number_of_input_channels" : self.number_of_channels,
            "number_of_output_classes" : self.number_of_output_classes,
            "input_spacing" : self.median_spacing
        }

        # In case that we want to export it
        if export_path is not None:
            # Modify name if dir is provided
            if not export_path.endswith(".json"):
                assert os.path.exists(export_path), \
                    f"Path for the export does not exists. Please check path: {export_path}!"
                export_path = os.path.join(export_path, "SNNUnet_fingerprint.json")

            # Dump it
            with open(export_path, "w") as _json_file:
                json.dump(_output_dict, _json_file, indent=4)

        # Return
        return _output_dict
    