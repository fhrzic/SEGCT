"""
Script which formats data based on the provided fingerprint.
It transfers everything to numpy and provides location files useful later on.
"""

import argparse
import os
import sys
import shutil

# Ensure the project directory is in the path (adjust as necessary)
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(current_file_path)
sys.path.append(main_dir)

from Utils.dataset_export import DatasetExport
from DatasetFingerprint.split_data_from_ratio import main as split_ratio_main
from DatasetFingerprint.generate_dataset_fingerprint import main as generate_fingerprint_main
from DatasetPreprocessing.dataset_formating import main as dataset_formating_main

def split_dataset_based_on_ratio( 
    input_dir: str,
    output_dir: str,
    ratio: float = 0.80,
    seed: int = 1993):
    """
    This function splits dataset into three subsets. Please see split_data_from_ratio.py script for all the info

    """
    split_ratio_main( 
        input_dir =input_dir,
        output_dir =output_dir,
        ratio = ratio,
        seed = seed)
    
def generate_fingerprint(
        volume_dir_path: str, 
        segmentations_dir_paths: str, 
        labels_mapping_dict_path: str,
        export_path: str,
        num_processes: int = None):
    """
    Function which generates fingerprint from training set. Please see generate_dataset_fingerprint.py for
    more info!.
    """
    generate_fingerprint_main(
        volume_dir_path =  volume_dir_path, 
        segmentations_dir_paths = segmentations_dir_paths, 
        labels_mapping_dict_path = labels_mapping_dict_path,
        export_path = export_path,
        num_processes = num_processes
    )

def dataset_formating(
         volumes_path: str,
         segmentation_path: str,
         fingerprint_path: str,
         export_dir_path: str,
         normalization_method: str = "F",
         number_of_processes: int = None,
         number_of_samples_per_class: int = None):
    """
    Function which format, crop, normalize and export the dataset in a form suitable for neural network
    training. For mor info see dataset_formating.py.
    """
    dataset_formating_main(
        volumes_path = volumes_path,
        segmentation_path = segmentation_path,
        fingerprint_path = fingerprint_path,
        export_dir_path = export_dir_path,
        normalization_method = normalization_method,
        number_of_processes = number_of_processes,
        number_of_samples_per_class = number_of_samples_per_class)

def cli():  
    """
    Command-line interface: parse arguments and call main(). It also create export dir
    and populates it!
    """
    _parser = argparse.ArgumentParser(
        description="Full pipeline which includes Dataset splitting with RATIO method, Fingerprinting, and Normalization"
    )
    _parser.add_argument(
        "--input_dir_path",
        dest="input_dir",
        required=True,
        help="Path to the directory which contains following directories: training_data, validation_data, and test_data (each of them has img and seg subdirs)."
    )

    _parser.add_argument(
        "--export_path",
        dest="export_dir_path",
        required=True,
        help="Path to the directory where data will be exported. IMPORTANT: OVERRIDE!"
    )

    _parser.add_argument(
        "--labels_path",
        dest="labels_path",
        required=True,
        help="Path to the labels.json folder. It must be provided! See readme_fingerprint for more details!"
    )

    _parser.add_argument(
        "--proc",
        dest="num_processes",
        type = int,
        required=False,
        help="Number of proceses for multiprocessing. Leave blank for no multiprocessing."
    )

    _args = _parser.parse_args()
    
    # Ok first we split the dataset
    print("Working on splitting dataset in ratio!")
    split_dataset_based_on_ratio(input_dir = _args.input_dir,
                                 output_dir= _args.export_dir_path,
                                 ratio = 0.8,
                                 seed = 1993)
    shutil.copy(_args.labels_path, _args.export_dir_path)

    # Fingerprint time
    print("Calculating fingerprint from the training set!.")
    generate_fingerprint(
        volume_dir_path = os.path.join(_args.export_dir_path, "test_data", "img"),
        segmentations_dir_paths = os.path.join(_args.export_dir_path, "test_data", "seg"),
        labels_mapping_dict_path = _args.labels_path,
        export_path = os.path.join(_args.export_dir_path, "training_fingerprint.json"),
        num_processes = _args.num_processes
    )

    # Generating numpys
    print("Exporting Dataset to NPYS!")
    for _name in ["test_data", "validation_data", "training_data"]: 
        # Verbose
        print(f"Working on: {_name}")
        dataset_formating(volumes_path = os.path.join(_args.export_dir_path, _name, "img"),
                          segmentation_path = os.path.join(_args.export_dir_path, _name, "seg"),
                        fingerprint_path = os.path.join(_args.export_dir_path, "training_fingerprint.json"),
                        export_dir_path = os.path.join(_args.export_dir_path, "formated_data", _name),
                        normalization_method = "F",
                        number_of_processes = _args.num_processes,
                        number_of_samples_per_class = 10000)

if __name__ == "__main__":
    cli()