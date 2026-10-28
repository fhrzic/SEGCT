"""
Scripts for splitting data from xlsx into test and train
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

def main(input_xlsx_path: str = None,
         export_dir_path:str = None):
    """
    Main script for splitting files into folders for train test and valid

    Args:
        * input_xlsx_path, str, path to the input xlsx file. In the dir where the xlsx file is
        located it also must have img and seg directories:
        some_dir:
            --> img
                --> <image name>.nii.gz
            --> seg
                --> <segmentation name>.nii.gz
            --> <name>.xlsx

        * export_dir_path, path to the export dir where all the folders gonna be created. The 
        export_dir_path will be created so be carefull to not override something important. In 
        the export path following dir structure will be created:
        export_dir_path:
            training_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz
            
            validation_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz
            test_data:
                    --> img
                        --> <image name>.nii.gz
                    --> seg
                        --> <segmentation name>.nii.gz
    """
    # Read xlsx
    _xlsx_df = pd.read_excel(input_xlsx_path)

    # Create dirs    
    _dir_name = os.path.dirname(input_xlsx_path)
    _images_dir = os.path.join(_dir_name, "img")
    _segmentation_dir = os.path.join(_dir_name, "seg")
    # Remove the directory if it exists
    if os.path.exists(export_dir_path):
        shutil.rmtree(export_dir_path)  # Removes the directory and its contents

    # Create a new directory
    os.makedirs(export_dir_path)

    # Create export dirs
    _train_dir = os.path.join(export_dir_path, "training_data")
    _validation_dir = os.path.join(export_dir_path, "validation_data")
    _test_dir = os.path.join(export_dir_path, "test_data")
    os.makedirs(_train_dir)
    os.makedirs(_validation_dir)
    os.makedirs(_test_dir)
    _training_seg = os.path.join(_train_dir, "seg")
    _training_ct = os.path.join(_train_dir, "ct")
    _valid_seg = os.path.join(_validation_dir, "seg")
    _valid_ct = os.path.join(_validation_dir, "ct")
    _test_seg = os.path.join(_test_dir, "seg")
    _test_ct = os.path.join(_test_dir, "ct")
    os.makedirs(_training_seg)
    os.makedirs(_training_ct)
    os.makedirs(_valid_seg)
    os.makedirs(_valid_ct)
    os.makedirs(_test_seg)
    os.makedirs(_test_ct)

    for _index, _row in tqdm(_xlsx_df.iterrows(),
                             total = len(_xlsx_df),
                             desc = "Files"):
        if _row["Set"] == "test":
            shutil.copy(os.path.join(_images_dir, _row["Image"]), 
                        os.path.join(_test_ct, _row["Image"]))
            shutil.copy(os.path.join(_segmentation_dir, _row["Segmentation"]), 
                        os.path.join(_test_seg, _row["Segmentation"]))
        if _row["Set"] == "train":
            shutil.copy(os.path.join(_images_dir, _row["Image"]), 
                        os.path.join(_training_ct, _row["Image"]))
            shutil.copy(os.path.join(_segmentation_dir, _row["Segmentation"]), 
                        os.path.join(_training_seg, _row["Segmentation"]))
        if _row["Set"] == "valid":
            shutil.copy(os.path.join(_images_dir, _row["Image"]), 
                        os.path.join(_valid_ct, _row["Image"]))
            shutil.copy(os.path.join(_segmentation_dir, _row["Segmentation"]), 
                        os.path.join(_valid_seg, _row["Segmentation"]))    
    print("DONE")
   
def cli():
    """
    Command-line interface: parse arguments and call main().
    """
    _parser = argparse.ArgumentParser(
        description="Process an XLSX file and export results to a specified directory."
    )
    _parser.add_argument(
        "--input",
        dest="input_xlsx_path",
        required=True,
        help="Path to the input XLSX file (e.g., /FullBody/data/MAPPING.xlsx)"
    )
    _parser.add_argument(
        "--export",
        dest="export_dir_path",
        required=True,
        help="Path to the export directory (e.g., /Desktop/MyProject/Data)"
    )
    _args = _parser.parse_args()
    # Call main
    main(input_xlsx_path=_args.input_xlsx_path, export_dir_path=_args.export_dir_path)

if __name__ == "__main__":
    cli()
    #_input_xlsx_path = "/mnt/SSD/Franko/Franko_segmentation/FullBody/data_corrected/MSK.xlsx"
    #_export_dir_path = "/mnt/SSD/Franko/Franko_segmentation/FullBody/data/"
    #main(input_xlsx_path = _input_xlsx_path,
    #     export_dir_path = _export_dir_path)
