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

def main(volumes_path: str,
         segmentation_path: str,
         fingerprint_path: str,
         export_dir_path: str,
         normalization_method: str = "F",
         number_of_processes: int = None,
         number_of_samples_per_class: int = None):
    """
    Main function for filtering and exporting data.

    Args:
        * volumes_path, str, Path to the CT volumes directory containing nifti files.
        * fingerprint_path, str, Path to the fingerprint file obtained by FingerprintGenerationScript.py.
        * export_dir_path, str, Path to the directory where data will be exported. IMPORTANT: OVERRIDE!
        * number_of_processes, int, Number of processes for multiprocessing. Default is None.
        * normalization_method, str, Normalization method to use ("F", "M", or "V"). Default is "F".
        * number_of_samples_per_class, int, number of samples for each class which will be stored in .npz files 
        (properties) for each case. If set to None, everything will be stored.

    """
    # Create DatasetExport object
    _d_export = DatasetExport(CT_files = volumes_path,
                             seg_maps = segmentation_path,
                             normalization_fingerprint_json = fingerprint_path,
                             normalization_method = normalization_method,
                             number_of_processes = number_of_processes,
                             number_of_samples_per_class = number_of_samples_per_class)
    # Export the data
    _d_export.export_data(dir_to_export_path = export_dir_path)

    # Export fingerprint
    _d_export.export_normalization_fingerprint(dir_to_export_path = export_dir_path)

def cli():  
    """
    Command-line interface: parse arguments and call main(). It also create export dir
    and populates it!
    """
    _parser = argparse.ArgumentParser(
        description="Format data based on the provided fingerprint, convert to numpy, and generate location files."
    )
    _parser.add_argument(
        "--input_dir_path",
        dest="input_dir",
        required=True,
        help="Path to the directory which contains following directories: training_data, validation_data, and test_data (each of them has img and seg subdirs)."
    )

    _parser.add_argument(
        "--fingerprint_path",
        dest="fingerprint_path",
        required=True,
        help="Path to the fingerprint file obtained by FingerprintGenerationScript.py."
    )
    _parser.add_argument(
        "--export_path",
        dest="export_dir_path",
        required=True,
        help="Path to the directory where data will be exported. IMPORTANT: OVERRIDE!"
    )
    _parser.add_argument(
        "--proc",
        dest="number_of_processes",
        type=int,
        default=None,
        required=False,
        help="Number of processes for multiprocessing (default: None)."
    )

    _parser.add_argument(
        "--normalization_method",
        dest="normalization_method",
        choices=["F", "M", "V"],
        default="F",
        required=False,
        help="Normalization method to use: F, M, or V (default: F)."
    )

    _parser.add_argument(
        "--number_of_samples_per_class",
        type=int,
        dest="num_samples",
        default = None,
        required=False,
        help="Number of random samples per class stored in loc. If set to None, all will be stored "
    )

    _args = _parser.parse_args()
    
    # Remove the export directory if it exists, then create it
    if os.path.exists(_args.export_dir_path):
        shutil.rmtree(_args.export_dir_path)
    os.makedirs(_args.export_dir_path)

    # Now lets populate export dir
    for _name in ["test_data", "validation_data", "training_data"]: 
        # Verbose
        print(f"Working on: {_name}")
        # Extract paths
        _volumes_path = os.path.join(_args.input_dir, _name, "img")
        _segmentations_path = os.path.join(_args.input_dir, _name, "seg")
        _export_path = os.path.join(_args.export_dir_path, _name)
        main(volumes_path =_volumes_path,
            segmentation_path = _segmentations_path,
            fingerprint_path = _args.fingerprint_path,
            export_dir_path = _export_path,
            normalization_method = _args.normalization_method,
            number_of_processes = _args.number_of_processes,
            number_of_samples_per_class = _args.num_samples)

if __name__ == "__main__":
    cli()