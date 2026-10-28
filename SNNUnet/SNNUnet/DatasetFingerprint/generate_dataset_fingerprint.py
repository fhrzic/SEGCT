""" Main script for generating data fingerprint"""
import argparse
import os
import sys


# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(main_dir)

from DatasetFingerprint.Utils.utility_functions import DataFingerprint  # Import any other needed functions as required

def main(volume_dir_path: str, 
         segmentations_dir_paths: str, 
         labels_mapping_dict_path: str,
         export_path: str,
         num_processes: int = None):
    """
    Main function for building the data fingerprint. Our data is organised as follows:

    CT_Volumes_dir:
        --> ct__001.nii.gz
        --> ct__002.nii.gz
        --> ...
        --> ct__n.nii.gz
    
    CT_segmentations_dir:
        --> ct_seg_001.nii.gz
        --> ct_seg_002.nii.gz
        --> ...
        --> ct_seg_n.nii.gz
    
    labels.json (with two labels groups) (structure: "pixel_value": name)

    The idea is to extract a list of files and then work with these files (obtain all important data). 
    For that purpose, we have created the class DataFingerprint which works on the described structure.
    It also supports input as a list of files and labels_mapping_dict as a dict instead of a list. 
    If a list is provided, it must have corresponding values (0 index in ).

    Args:
        * volume_dir_path, str, Path to the directory containing CT volumes (nii.gz).
        * segmentations_dir_paths, str, Path to the directory containing segmentation maps (nii.gz).
        * labels_mapping_dict_path, str, Path to the JSON file containing labels mapping.
        * export_path, str, Path to the export file where the fingerprint will be saved.
        * num_processes, int, number of processes for multiprocessing. Default is None which means 
        no multiprocessing
    """

    _my_data_fingerprint = DataFingerprint(
        CT_files=volume_dir_path,
        seg_maps=segmentations_dir_paths,
        labels=labels_mapping_dict_path,
        ignore_label=-1,
        number_of_processes=num_processes
    )
    _my_data_fingerprint.export_fingerprint(export_path=export_path)

def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="Generate a data fingerprint from CT volumes, segmentation maps, and a labels mapping JSON file."
    )
    _parser.add_argument(
        "--volumes",
        dest="volume_dir_path",
        required=True,
        help="Path to the directory containing CT volumes (nii.gz)."
    )
    _parser.add_argument(
        "--segmentations",
        dest="segmentations_dir_paths",
        required=True,
        help="Path to the directory containing segmentation maps (nii.gz)."
    )
    _parser.add_argument(
        "--labels",
        dest="labels_mapping_dict_path",
        required=True,
        help="Path to the JSON file containing labels mapping."
    )
    _parser.add_argument(
        "--export",
        dest="export_path",
        required=True,
        help="Path to the export file where the fingerprint will be saved."
    )

    _parser.add_argument(
        "--proc",
        dest="num_processes",
        type = int,
        required=False,
        help="Number of proceses for multiprocessing. Leave blank for no multiprocessing."
    )
    _args = _parser.parse_args()
    
    main(
        volume_dir_path = _args.volume_dir_path,
        segmentations_dir_paths = _args.segmentations_dir_paths,
        labels_mapping_dict_path = _args.labels_mapping_dict_path,
        export_path = _args.export_path,
        num_processes = _args.num_processes
    )

if __name__ == "__main__":
    cli()