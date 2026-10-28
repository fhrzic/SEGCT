""" Class for creating custom batch sampler"""
import random
import sys
import os
from torch.utils.data import Sampler
import numpy as np

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json
from Configurations.config_data import SamplerConfig

class ValidationSampler(Sampler):
    """
    Sampler which samples only foreground patches from given samples. It selects a
    sample, caluclate number of patches for that sample based on the desired number of patches spreaded equaly
    accross the available samples and yields them back as ct and segs concatenated. Then it does same for
    the next case.
    """
    def __init__(self, 
                 sampler_config: SamplerConfig = None):
        """
        Init method of a sampler. This sampler makes sure of applaying following conditions:
            1.) It requires to know number of total samples in dataset, and how many samples is
            required to be sampled. Because it 
        """
        super().__init__()

        # Random seed
        random.seed(1993)
        np.random.seed(1993)
        ## Obtain necessary data
        #Dataset
        self.data_source = sampler_config.dataset

        # Number of targeted samples. We know how much we need to sample for each case
        self.number_of_samples = int(sampler_config.number_of_samples / len(sampler_config.dataset))
        self.number_of_samples = max(self.number_of_samples, 1)

        # Labels
        self.classes_per_case = self.data_source.get_cases_and_classes()
                                              
    def __iter__(self):
        # For each sample in dataset
        for _case_id in range(len(self.data_source)):
            # We now need to sample the calculated amount of foreground voxels from
            # the selected index

            # Create mega-batch
            _batch = []
            
            # Sample exact number of batches for this case
            for _index in range(self.number_of_samples):
                # Now we sample

                # Get classes in case
                _case_classes = self.classes_per_case[_case_id]
                
                # Remove background 
                if "0" in _case_classes:
                    _case_classes.remove("0")
                
                # Chose randomly class from the classes in case
                _class = np.random.choice(a = _case_classes, size = 1)
                
                # Add case
                _batch.append({"index": _case_id, 
                              "class": _class[0]})
            yield _batch

    def __len__(self):
        # Number of batches in the dataset
        return len(self.data_source)