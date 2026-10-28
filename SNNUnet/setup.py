"""
Setup for cli scripts
"""
from setuptools import setup, find_packages

setup(
    name="SNNUnet",
    version="0.1.0",
    python_requires="==3.10.14",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "tqdm",
        "openpyxl",
        "einops",
        "SimpleITK",
        "scipy",        
        "torchio==0.19.6", #This can be upgraded, but there is an warning with torch so I am keeping it in the stable version
        "torch",
        "torchinfo",
        "accelerate",
        "matplotlib"
    ],
    entry_points={
        'console_scripts': [
            'split_data_from_xlsx = SNNUnet.DatasetFingerprint.split_data_from_xlsx:cli',                 # DATA MANIPULATION
            'split_data_from_ratio = SNNUnet.DatasetFingerprint.split_data_from_ratio:cli',               # DATA MANIPULATION
            'generate_dataset_fingerprint = SNNUnet.DatasetFingerprint.generate_dataset_fingerprint:cli', # DATA MANIPULATION
            'dataset_formating = SNNUnet.DatasetPreprocessing.dataset_formating:cli',                     # DATA MANIPULATION
            'create_dataset = SNNUnet.DatasetPreprocessing.create_dataset:cli',                           # DATA MANIPULATION
            'verify_dataloader = SNNUnet.Data.verify_dataloader:cli',                                     # DATALOADER PIPELINE TESTING
            'inspect_model = SNNUnet.Models.inspect_model:cli',                                           # MODEL TESTING
            'inspect_loss = SNNUnet.Losses.inspect_loss:cli',                                             # LOSS FUNCTION TESTING
            'test_training = SNNUnet.Training.manual_training:cli',                                       # TRAINING TESTING
            'run_training_from_config = SNNUnet.Training.configuration_based_training:cli',               # TRAINING FROM CONFIGS   
            'run_preset_pipeline = SNNUnet.RunScripts.run_preset_pipeline:cli',                           # EXECUTIVE SCRIPT FOR TRAINING
            'run_argument_pipeline = SNNUnet.RunScripts.run_argumented_pipeline:cli',                     # EXECUTIVE SCRIPT FOR TRAINING
            'run_inference = SNNUnet.Inference.perform_inference:cli',                                    # EXECUTIVE SCRIPT FOR INFERENCE
            'run_statistics = SNNUnet.Inference.inference_statistics_evaluation:cli',                     # EXECUTIVE SCRIPT FOR INFERENCE
            'run_inference_from_dir = SNNUnet.Inference.run_inference_from_dir:cli',                      # EXECUTIVE SCRIPT FOR INFERENCE
            'archive_model = SNNUnet.Inference.archive_model:cli',                                        # MODEL ARCHIVING SCRIPT
        ],
    },
)

