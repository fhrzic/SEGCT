""" This script do inference on the provided dir"""

import sys
import os
import time
import json
import argparse
from tqdm import tqdm
# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)


from Configurations.config_inference import InferenceConfig
from Inference.inference_container import *

def main(
        input_dir_ct: str = None,
        input_dir_seg: str = None,
        output_dir: str = None,
        normalization_fingerprint: str = None,
        data_fingerprint: str = None,
        model_config: str = None,
        model_weights: str = None,
        device: str = "cuda:0"):
    """
    Main script for inference. It conducts inference on all data found in the provided
    input_dir_ct. It exports inferences in the output_dir. The inference is done by provided 
    model (model_config and model_weights), while for normalization of data path to normalization_fingerprint
    file which can be find in "formated" dir of the data generation.

    Args:
        * input_dir_ct, string, path to the dir which contains CT data. By CT data it supports either ".npy"
        (already normalized files) or "nifti" files which will undergo normalization.
        * input_dir_seg, string, path to the dir which contains segmentation masks. If provided, it will be
        transfered along with the prediction to the export dir. If not (it is None) it is fine - but then
        calculating statistics later on will not be possible.
        * output_dir, string, directory where all outputs of inference will be stored. If directory does not exists it
        will be created. If it already exists, the predictions will be appended to it.
        * normalization_fingerprint, string, path to the config file containing normalization details. Usually
        it is created in the same location where the "formated" data is exported.
        * data_fingerprint, string, path to the data fingerprint file. I know that it can be part of normalization
        fingeprint, but it is also possible that it is not!
        * model_config, string, path to the model config which is necessary to build the model. 
        * model_weights, string, path to the model's weights which will be loaded into the model for prediction.
        * device, string, device on which the inference will be performed. E.G. cpu, cuda:0, cuda:1...
    """
    # Build the inference config file
    # Define infernece params
    _inference_config = InferenceConfig()
    InferenceConfig.model_path = model_weights
    InferenceConfig.output_dir = output_dir
    InferenceConfig.device = device
    InferenceConfig.model_config = model_config #Or tupple (128,128,296)
    InferenceConfig.batch_size = 2
    InferenceConfig.normalization_fingerprint = normalization_fingerprint
    InferenceConfig.data_fingerprint = data_fingerprint
    InferenceConfig.overlap = 0.5
    InferenceConfig.verbose = True
    InferenceConfig.gaussian = True
    InferenceConfig.flip = False
    # Print
    print(_inference_config)

    # Create container
    _my_inference_container = InferenceContainer(inference_config = _inference_config) 

    # Set dirs
    _input_dir = input_dir_ct
    _input_segmentations = input_dir_seg

    # Check if it is dir, if not it means it is file so we perfrom inference
    # Only for that one file
    if os.path.isfile(_input_dir):
        _input_dir_list = [_input_dir]
        _input_seg_list = [_input_segmentations]
    else:
        # Obtain lists
        _input_dir_list = [os.path.join(_input_dir, _file) for 
                        _file in os.listdir(_input_dir)]
        if _input_segmentations is None:
            _input_seg_list = [None for _ in _input_dir_list]
        else:
            _input_seg_list = [os.path.join(_input_segmentations, _file) for 
                            _file in os.listdir(_input_segmentations)]
            # Sort
            _input_seg_list.sort()
        # Sort
        _input_dir_list.sort()
        
        
    # Output dict
    _times_dict = {}
    for _i in tqdm(range(len(_input_dir_list)), desc = "Files"):
        # Measure time
        _start_time = time.time()
  
        # Inference
        _my_inference_container.obtain_prediction(path_to_src = _input_dir_list[_i],
                                            path_to_segmentation = _input_seg_list[_i],
                                            save_original = False,
                                            save_npy = False)

        _end_time = time.time()
        _times_dict[_input_dir_list[_i]] = _end_time-_start_time

    # Print
    print(_times_dict)
    
    # Export dictionary to a JSON file
    with open(os.path.join(_inference_config.output_dir, "times.json"), 'w') as _json_file:
        json.dump(_times_dict, _json_file, indent=4)  # indent=4 makes it more readable
    
    # Export inference config to the directory
    _inference_config.export_to_json(export_path = os.path.join(_inference_config.output_dir, "inference_config.json"))

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="It only needs output dir and input dir... nothing else"
    )
    _parser.add_argument(
        "--input_dir_ct",
        dest="input_ct_dir",
        required=True,
        help="Path to the input directory which contains CTs"
    )

    _parser.add_argument(
        "--input_dir_seg",
        dest="input_seg_dir",
        required=False,
        default=None,
        help="Path to the input directory which contains Segmentations"
    )

    _parser.add_argument(
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Path to the output directory. (WARNING: This directory will be created and any existing content will be deleted.)"
    )
    
    _parser.add_argument(
        "--normalization_fingerprint",
        dest="normalization_fingerprint",
        required=True,
        help="Path to the JSON file containing normalization fingerprint"
    )

    _parser.add_argument(
        "--data_fingerprint",
        dest="data_fingerprint",
        required=True,
        help="Path to the JSON file containing dataset fingerprint"
    )

    _parser.add_argument(
        "--model_config",
        dest="model_config",
        required=True,
        help="Path to the JSON file containing models' config"
    )

    _parser.add_argument(
        "--model_weights",
        dest="model_weights",
        required=True,
        help="Path to the pth file containing model's weights"
    )

    _parser.add_argument(
        "--device",
        dest="device",
        required=True,
        default ="cpu",
        help="Device on which the inference will be performed."
    )
       
    _args = _parser.parse_args()

    # Generate subdirectories
    
    main(
        input_dir_ct =_args.input_ct_dir,
        input_dir_seg = _args.input_seg_dir,
        output_dir =_args.output_dir,
        normalization_fingerprint= _args.normalization_fingerprint,
        data_fingerprint= _args.data_fingerprint,
        model_config = _args.model_config,
        model_weights = _args.model_weights,
        device = _args.device
    )
  
if __name__ == "__main__":
    main()
