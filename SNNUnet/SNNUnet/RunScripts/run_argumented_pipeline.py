"""
For lazy workers, this is the script to call
"""
import sys
import os
import shutil
import argparse

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from DatasetFingerprint.split_data_from_ratio import main as split_main
from DatasetFingerprint.generate_dataset_fingerprint import main as fingerprint_main
from DatasetPreprocessing.dataset_formating import main as dataset_main
from Training.manual_training_with_arguments import main as training_main

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="It only needs output dir and input dir... nothing else"
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
        "--labels",
        dest="labels_mapping_dict_path",
        required=True,
        help="Path to the JSON file containing labels mapping."
    )

    _parser.add_argument(
        "--epochs",
        dest="epochs",
        default = 1000,
        type = int,
        required=False,
        help="Number of epochs to be trained."
    )

    _parser.add_argument(
        "--patch_shape",
        dest="patch_shape",
        type=int,
        nargs=3,
        default=[128, 128, 196],
        required=False,
        help="Desired models patch_shape, i.e. 128 128 196 (3 integers) is (128x128x196)."
    )

    _parser.add_argument(
        "--normalization_method",
        dest="normalization_method",
        required=False,
        help="Desired normalization method: F, M or V"
    )

   
    _args = _parser.parse_args()

    # Generate subdirectories
    split_main(
        input_dir=_args.input_dir,
        output_dir=_args.output_dir,
        ratio = 0.7,
        seed = 1993
    )

    fingerprint_main(
        volume_dir_path = os.path.join(_args.output_dir, "training_data", "img"),
        segmentations_dir_paths = os.path.join(_args.output_dir, "training_data", "seg"),
        labels_mapping_dict_path = _args.labels_mapping_dict_path,
        export_path = os.path.join(_args.output_dir, "training_fingerprint.json"),
        num_processes = 10
    )
    
    
    # Generating numpys
    print("Exporting Dataset to NPYS!")
    for _name in ["test_data", "validation_data", "training_data"]: 
        # Verbose
        print(f"Working on: {_name}")
        dataset_main(volumes_path = os.path.join(_args.output_dir, _name, "img"),
                        segmentation_path = os.path.join(_args.output_dir, _name, "seg"),
                        fingerprint_path = os.path.join(_args.output_dir, "training_fingerprint.json"),
                        export_dir_path = os.path.join(_args.output_dir, "formated_data", _name),
                        normalization_method = _args.normalization_method,
                        number_of_processes = 10,
                        number_of_samples_per_class = 10000)

    
    # Traning
    training_main(export_dir_path = os.path.join(_args.output_dir, "TRAINING"),
         device = "cuda:0",
         path_to_data_fingerprint = os.path.join(_args.output_dir, "training_fingerprint.json"),
         data_path =  os.path.join(_args.output_dir, "formated_data"),
         epochs = _args.epochs,
         patch_shape = _args.patch_shape
    )

if __name__ == "__main__":
    cli()