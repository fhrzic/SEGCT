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

class NNUnetSampler(Sampler):
    """
    Sampler which samples exactly batch number of cases, and then divides them into 
    two groups: fg required, and bg required. For fg required it seeks for the the random
    class in the case. For bg, well nothing :)
    """
    def __init__(self, 
                 sampler_config: SamplerConfig = None):
        """
        Init method of a sampler. This sampler makes sure of applaying following conditions:
            1.) At least %fg_sample_ratio with min of 1 samples in batch are sampled with 
            forground present
            2.) When sampling a foreground, special care is taken that the classes are balanced. 
            This means that in in fg_sample_ratio of cases the unifrom distirbution of classes 
            will be present. In ~fg_sample_ratio cases, well - it-is-as-it-is random patach 
            of a random sample.
            3.) OVERRIDE OF UNIFORM SAMPLING. If for some reason we dont want to sample from uniform
            distribution, in that case in SamplerConfig "class_sampling_dict" parametar must contain
            valid dict where key is a valid label and value is it sampling probability (all values 
            sum to 1). Additionaly, "class_sampling_dict" can be string which is then path to the
            fingerprint. Then the dict is caluclated on from FG ratios. In this case all labels in the
            dataset must be provided!
        """
        super().__init__()

        # Random seed
        random.seed(1993)
        np.random.seed(1993)
        ## Obtain necessary data
        #Dataset
        self.data_source = sampler_config.dataset

        # Batch size
        self.batch_size = sampler_config.batch_size

        # Labels
        self.classes_per_case = self.data_source.get_cases_and_classes()

        # FG sampling ratio
        self.number_of_fg_cases = max(1, int(sampler_config.fg_sample_ratio*self.batch_size))
                                              
    def __iter__(self):
        # Custom logic to yield batches of indices. Logic described in Init function
        while True:
            _batch = []
            
            # Select random cases for batch
            _cases_ids = np.random.choice(a = list(self.classes_per_case.keys()), size = self.batch_size, replace = True)

            # Now we need to select samples for foreground
            for _i in range(self.number_of_fg_cases):
                # Get classes in case
                _case_classes = self.classes_per_case[_cases_ids[_i]]
                
                # Remove background 
                if "0" in _case_classes:
                    _case_classes.remove("0")
                
                # Chose randomly class from the classes in case
                _class = np.random.choice(a = _case_classes, size = 1)
                
                # Add case
                _batch.append({"index": _cases_ids[_i], 
                              "class": _class[0]})
            for _i in range(self.number_of_fg_cases, self.batch_size):
                _batch.append({"index": _cases_ids[_i], 
                              "class": "-1"})
            random.shuffle(_batch)
            yield _batch


    def __len__(self):
        # Number of batches in the dataset
        return float('inf')
    
