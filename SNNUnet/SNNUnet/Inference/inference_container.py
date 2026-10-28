""" Main inference starting point. This serves for the whole inference"""
import gc
import os
import sys
import shutil
from itertools import combinations
import numpy as np
import SimpleITK as sitk
import torch
from typing import List, Tuple
from tqdm import tqdm
import json

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)


from Configurations.config_inference import InferenceConfig
from Configurations.config_model import ModelConfig
from Models.VNet.V_net import *
from Models.SNNUnet.SNNUnet import *
from Inference.inference_utility_functions import *
from DatasetPreprocessing.Utils.normalization import *
from Models.SNNUnet.autograd_and_gradient_clipping import DummyContext
from Models.SNNUnet.building_blocks import *


class InferenceContainer():
    """
    Class for inference of on the data. The idea is to load model, obtain data,
    do the inference and save segmentation map
    """
    def __init__(
        self,
        inference_config: InferenceConfig = None):
        """
        Method which acceptes inference config as an argument and based on it sets up the whole
        container.

        Args:
            * inference_config, InferenceConfig, config file for the inference. 
            For more info visit config_inference file.
        """
        # Get verbose flag
        self.verbose = inference_config.verbose

        # Make config globaly available
        self.inference_config = inference_config

        # Obtain number of classes
        self.num_classes = self.__obtain_number_of_classes()

        # Obtain model
        if self.verbose:
            print(f"Loading model saved in {inference_config.model_path}"
                  f"to device: {inference_config.device}!")
        self.model = self.__obtain_model(
            model_path = inference_config.model_path,
            device = inference_config.device)

        # Output dir - create if it does not exists
        if not os.path.exists(inference_config.output_dir):
            if self.verbose:
                print(f"Creating output directory {inference_config.output_dir}!")
            os.makedirs(inference_config.output_dir)
        else:
            if self.verbose:
                print(f"Output directory {inference_config.output_dir}"
                      "already exists, appending to it!")
        self.output_dir = inference_config.output_dir

        # Enable fliping during the prediction
        if inference_config.flip is True:
            self.mirror_axes =  self.__obtain_mirror_axes()
        else:
            self.mirror_axes = False
        
        # Obtain patch size
        self.patch_shape = self.__obtain_patch_shape()

    def __obtain_number_of_classes(
        self)->int:
        """
        Method which obtains number of classes from fingeprprint json.

        Returns:
            * int, a single number represenitng number of classes
        """
        # Read data and obtrain json
        try:
            _dictionaries = read_data_fingerprint_json(self.inference_config.data_fingerprint)
            _classes = _dictionaries["LABELS_MAPPING"]

        except ValueError as _exc:
            raise ValueError(
                "Something went wrong in class obtaining."
                f"Could not obtain data from {self.inference_config.data_fingerprint}!") from _exc
        # Return
        return len(list(_classes.keys())) - 1 # -1 is because of ignore label

    def __obtain_model(
        self,
        model_path: str = None,
        device: str = "cpu")->torch.nn.Module:
        """
        Method which configures and create model

        Args:
            * model_path, string, path to the model which is going to be loaded into memory
            * device, string, string describing where the model will be set
        
        Returns:
            * model obtained by torch.nn.Module
        """
        # Load model and place it on the device
        _model = torch.load(model_path, weights_only=False, map_location = device)

        # Return
        return _model


    def __obtain_patch_shape(self,
                            model_config_path = (128,128,192), 
                            )->List[Tuple[int]]:
        """
        Method which extracts models' patch size. If string is provided
        then it is assumed that path to json containing model's config is
        given. Else, it is expected to be tuple.

        Args:
            * model_config_path, string or Tuple, if string is provided then
            it must point to the model config json data from which the input shape 
            (patch size) is extracted. If not that, then tuple with exact size in
            apropriate channel order must be provided. E.G. (112, 112, 192)
        """
        # Check if it is a string or tupple
        if isinstance(model_config_path, str):
            # String -->load json
            _model_config = ModelConfig()
            _model_config.import_from_json(import_path = model_config_path)
            return _model_config.input_shape
        else:
            # We assume tuple is provided
            return model_config_path

    def __obtain_mirror_axes(
        self)->List[Tuple[int]]:
        """
        Method which returns all combinations of axes required for mirroring. It support only 3D
        
        Returns:
            * list of tuples where each tuple contain integers representing axes to be mirrored.
            For instance [(2),(3,4),(2,3,4)] --> axes 2, 3 and 4 in second case and all axes in 
            the final case should be mirrored. Axes 0 and 1 are batch and channel.
        """
        # Generate mirror axes
        _mirror_axes = [2, 3, 4]
        
        # Obtain axes combinatios
        _axes_combinations = [
            _c for _i in range(len(_mirror_axes)) for _c in combinations(_mirror_axes, _i + 1)
        ]
        print(_axes_combinations)

        # Return
        return _axes_combinations

    def __obtain_background_label(
        self)->int:
        """
        Method for obtaining the background label value
        """
        # Read data and obtrain json
        try:
            _dictionaries = read_data_fingerprint_json(self.inference_config.data_fingerprint)
            _classes = _dictionaries["LABELS_MAPPING"]
            _label_value = [_key for _key, _value in _classes.items() if _value == "background"][0]

        except ValueError as _exc:
            raise ValueError(
                "Something went wrong in background value obtaining." 
                f"Could not obtain data from {self.inference_config.data_fingerprint}!") from _exc

        # Return
        return float(_label_value)

    def __model_predict(
        self, 
        input_data:np.ndarray = None)->np.ndarray:
        """
        Method which predicts segmentation based on given input data.

        Args:
            * input_data, np.ndarray, data which is used as the input to the model
        
        Returns:
            * np.ndarray which represents models prediction in the same shape as the input data
        """
        # Get array into shape and split it into patches
        # Go to tensor
        _ct_tensor = torch.from_numpy(input_data)
        _ct_tensor = _ct_tensor.to(torch.float32)
        
        # Pad it
        _ct_tensor, _padding_offset, _original_shape = symetrical_padding(
            volume = _ct_tensor,
            desired_shape = self.patch_shape,
            patch_shape = self.patch_shape,
            divisible_by_patch = True,
            padding_value = self.__obtain_background_label())
        
        # Obtain locations
        _ct_patches, _patches_locations = split_volume_into_patches_no_padding(
            volume = _ct_tensor, 
            patch_shape = self.patch_shape,
            overlap = self.inference_config.overlap)
        _shape = _ct_tensor.shape

        # Add channel
        _ct_patches = _ct_patches.unsqueeze(1)

        # Split it to "batches"
        _ct_splits = torch.split(_ct_patches, self.inference_config.batch_size, dim=0)

        # Temporary prediction storage
        _tmp_predictions = []

        # Predict
        self.model.eval()
        with torch.autocast(device_type=_ct_patches.device.type, enabled=True) \
                if _ct_patches.device.type == 'cuda' else DummyContext():
            for _split in tqdm(_ct_splits):
                # Tranfer to device and do prediction
                _split = _split.to(self.inference_config.device)

                with torch.no_grad():
                    _prediction = self.model(_split)

                # If deep supervision (aux output) is on in model, model will produce multiple outputs
                if isinstance(_prediction, list):
                    _prediction = _prediction[-1]

                # Important
                _prediction = _prediction.detach()

                # If mirroring is on this process must be repeated for each mirror-axis
                if self.mirror_axes:
                    for _axes in self.mirror_axes:
                        # Flip - predict then unflip
                        # Predict on flipped torch tensor
                        with torch.no_grad():
                            _flip_pred = self.model(torch.flip(_split, _axes))
                        
                        # If deep supervision was on, models produces multiple outputs
                        if isinstance(_flip_pred, list):
                            _flip_pred = _flip_pred[-1]
                        
                        # Flip back and detach (saves memory)
                        _flip_pred = torch.flip(_flip_pred, _axes).detach()
                        _prediction += _flip_pred
                    # Average
                    _prediction /= (len(self.mirror_axes) + 1)

                # Necessary to avoid OOM
                _prediction = _prediction.to("cpu")
                _prediction = _prediction.half()
                _split = _split.to("cpu")
                _tmp_predictions.append(_prediction)

        # Release GPU
        #del self.model
        del _ct_splits
        del _ct_patches
        gc.collect()
        torch.cuda.empty_cache()

        # Merge predictions
        _tmp_predictions = torch.cat(_tmp_predictions, dim = 0)

        # Reduce precision
        _tmp_predictions = _tmp_predictions.half()

        # Reconstruct volume
        _reconsturcted_multiclass_segmentation = reconstruct_volume_from_patches(
            patches = _tmp_predictions, 
            patch_locations = _patches_locations,
            volume_shape = _shape,
            patch_shape = self.patch_shape,
            gaussian = self.inference_config.gaussian,
            number_of_classes = self.num_classes,
            device = self.inference_config.device)

        # Calculate argmax
        _original_shaped_segmentation = torch.argmax(
            _reconsturcted_multiclass_segmentation, dim = 0)
        _original_shaped_segmentation = _original_shaped_segmentation.squeeze(0)

        # Remove padding
        _original_shaped_segmentation = unpad_volume(volume = _original_shaped_segmentation,
                                                      padding_offset = _padding_offset,
                                                      original_volume_shape = _original_shape)
        # Return
        return _original_shaped_segmentation.cpu().numpy()

    def obtain_prediction(
        self,
        path_to_src: str = None,
        path_to_segmentation: str = None,
        save_original = True,
        save_npy = True):
        """
        Method which run inference of the model on the provided file. It can be either nii.gz or
        npy. In case of nii.gz, the npy will be generated. Segmentation map will be exported to 
        the directory provided in inference_config of the class. The name of segmentation map 
        will be derived from the provided path_to_file as its file name with extension of "_seg". 
        It will also be exported as nii.gz version of seg map if nii.gz was provided as the input

        Args:
            * path_to_src, string, string to the file path. It is either nii.gz or npy.
            * path_to_segmentation, string, string representing path to the segmentation in 
            nii.gz format (it will be saved under that extension!!!)
            * save_original, boolean, this flag singals if the original input data will be 
            transfered to the new location. It is convinient to have all in one place
            * export_npy, boolean, if true, npy will be exported.
        
        
        Returns:
            * it exports segmentation maps in nii.gz and npy. format (if the nii.gz was provided). 
        """
        # Check for file extension validity
        _file_name_full = os.path.basename(path_to_src)
        if ".npy" in _file_name_full:
            _file_name = _file_name_full.split(".npy")[0]
            _file_extension = "npy"
        else:
            # .nii.gz
            _file_name = _file_name_full.split(".nii.gz")[0]
            _file_extension = "nii.gz"

        assert _file_extension in ["npy", "nii.gz"], \
            f"Wrong extension! Expected .npy or .nii.gz but got {_file_extension}!"

        # Create save folder for the case (THIS OVERRIDE)
        _case_dir = os.path.join(self.inference_config.output_dir, _file_name)
        if os.path.exists(_case_dir):
            # Remove the directory and all its contents
            shutil.rmtree(_case_dir)
        # Create the directory
        os.makedirs(_case_dir)

        # In case of numpy
        if _file_extension == "npy":
            # Load
            _ct_array = np.load(path_to_src)
        else:
            # In case of CT, extract npy
            if self.verbose: 
                print(f"Transfering nii.gz to npy for case {_file_name}!")
            _ct_array = transfer_ct_to_npy(
                path_to_ct = path_to_src,
                normalization_fingerprint_path = self.inference_config.normalization_fingerprint,
                data_fingerprint_path = self.inference_config.data_fingerprint)

            # Save npy
            if save_npy:
                if self.verbose:
                    print(f"Exporting npy for case {_file_name}!")
                np.save(os.path.join(_case_dir,"original.npy"), _ct_array)    

        # Obtain prediction
        _predicted_segmentation = self.__model_predict(input_data = _ct_array)

        # Save segmentation as npy
        if save_npy:
            if self.verbose: 
                print(f"Exporting segmentation.npy for case {_file_name}!")
            np.save(os.path.join(_case_dir,"predicted_segmentation.npy"), _predicted_segmentation)

        # Save segmentation as nii.gz (available only if nii.gz is provided)
        if _file_extension == "nii.gz":
            _seg_map_nifti = transfer_to_sitkImg(
                data = _predicted_segmentation, original_ct_path = path_to_src)
            sitk.WriteImage(_seg_map_nifti, os.path.join(_case_dir,"predicted_segmentation.nii.gz"))

        # Save original 
        if save_original:
            if self.verbose: 
                print(f"Transfering orignial file for case {_file_name}!")
            shutil.copy(path_to_src, os.path.join(_case_dir, f"original.{_file_extension}"))

        # Save segmentation if provided
        if save_original and path_to_segmentation is not None:
            if self.verbose: 
                print(f"Transfering orignial segmentation for case {_file_name}!")
            shutil.copy(
                path_to_segmentation,
                os.path.join(_case_dir, f"original_segmentation.nii.gz"))
            
