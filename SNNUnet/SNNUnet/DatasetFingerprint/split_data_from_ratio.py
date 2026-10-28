""" Classical split of the data into train, test and valid directories """

import random
import os
import shutil
import argparse
from tqdm import tqdm

def main(
    input_dir: str,
    output_dir: str,
    ratio: float = 0.80,
    seed: int = 1993
):
    """
    Script for splitting train, validation, and test sets.

    Args:
        * input_dir (str): Path to the input directory containing subfolders 'img' and 'seg'.
        some_dir:
            --> img
                --> <image name>.nii.gz
            --> seg
                --> <segmentation name>.nii.gz
        * output_dir (str): Path to the output directory. WARNING: The output directory will 
        be created (existing content will be deleted). This is the outline 

        output_dir:    
            -->training_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz
            
            -->validation_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz
            -->test_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz

        * ratio (float): Ratio of data kept in the training set. The remaining data is divided equally 
        between validation and test sets.
        * seed (int): Seed for random number generator.
    """
    # Set seed
    random.seed(seed)

    # Obtain files
    _input_ct_dir = os.path.join(input_dir, "img")
    _input_seg_dir = os.path.join(input_dir, "seg")
    _ct_file_list = [os.path.join(_input_ct_dir, _f) for _f in os.listdir(_input_ct_dir)]
    _seg_file_list = [os.path.join(_input_seg_dir, _f) for _f in os.listdir(_input_seg_dir)]

    # Sort them so that corresponding files are aligned
    _ct_file_list.sort()
    _seg_file_list.sort()

    # Remove output directory if it exists, then create a fresh one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Calculate number of samples to transfer for validation and test
    _number_of_samples = int(len(_ct_file_list) * (1 - ratio))
    _number_of_samples = int(_number_of_samples / 2)
    # Ensure that it is at least 1
    _number_of_samples = max(_number_of_samples, 1)

    # Split indices and export data for each set
    _sampling_indices = list(range(len(_ct_file_list)))
    for _export_name in ["validation_data", "test_data", "training_data"]:
        if _export_name != "training_data":
            # Choose random indices for validation and test sets
            _indices = random.sample(_sampling_indices, _number_of_samples)
            _sampling_indices = [_item for _item in _sampling_indices if _item not in _indices]
        else:
            # Remaining indices go to the training set
            _indices = _sampling_indices

        # Create target directories
        _target_dir = os.path.join(output_dir, _export_name)
        os.makedirs(_target_dir)
        _output_ct_dir = os.path.join(_target_dir, "img")
        _output_seg_dir = os.path.join(_target_dir, "seg")
        os.makedirs(_output_ct_dir)
        os.makedirs(_output_seg_dir)

        # Copy files to the corresponding output directories
        for _idx in tqdm(_indices, total=len(_indices), desc=_export_name):
            # Copy CT file
            _ct_name = os.path.basename(_ct_file_list[_idx])
            shutil.copy(_ct_file_list[_idx], os.path.join(_output_ct_dir, _ct_name))
            # Copy segmentation file
            _seg_name = os.path.basename(_seg_file_list[_idx])
            shutil.copy(_seg_file_list[_idx], os.path.join(_output_seg_dir, _seg_name))

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="Split data into training, validation, and test sets based on a given ratio."
    )
    _parser.add_argument(
        "--input",
        dest="input_dir",
        required=True,
        help="Path to the input directory containing subfolders 'img' and 'seg'."
    )
    _parser.add_argument(
        "--output",
        dest="output_dir",
        required=True,
        help="Path to the output directory. (WARNING: This directory will be created and any existing content will be deleted.)"
    )
    _parser.add_argument(
        "--ratio",
        type=float,
        default=0.80,
        help="Ratio of data to be kept in the training set (default: 0.80). The remaining data is split equally between validation and test sets."
    )
    _parser.add_argument(
        "--seed",
        type=int,
        default=1993,
        help="Seed for random number generation (default: 1993)."
    )
    _args = _parser.parse_args()
    main(
        input_dir=_args.input_dir,
        output_dir=_args.output_dir,
        ratio=_args.ratio,
        seed=_args.seed
    )

if __name__ == "__main__":
    cli()
    #_input_xlsx_path = "/mnt/SSD/Franko/Franko_segmentation/FullBody/data_corrected/MSK.xlsx"
    #_export_dir_path = "/mnt/SSD/Franko/Franko_segmentation/FullBody/data/"
    #main(input_xlsx_path = _input_xlsx_path,
    #     export_dir_path = _export_dir_path)