""" Class for creating custom batch sampler"""
import random
import sys
import os
from torch.utils.data import Sampler
import numpy as np
from .config_dataloader import SamplerConfig

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json
from Configurations.config_data import SamplerConfig

class CustomBatchSampler(Sampler):
    """
    Sampler which samples the random patach for a random class.
    It mimics the NNUnet behaviour of data sampling but with special care
    that each class is sampled same number of times.
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
        self.labels = self.data_source.get_labels()

        # FG sampling ratio
        self.number_of_fg_cases = max(1, int(sampler_config.fg_sample_ratio*self.batch_size))

        # Distribution for sampling
        self.sampling_distribution = None
        if isinstance(sampler_config.class_sampling_dict, dict):
            # Dictionary is provided, full responsibility is on the user to provide correct dict
            self.sampling_distribution = sampler_config.class_sampling_dict
        
        if isinstance(sampler_config.class_sampling_dict, str):
            # Path to fingerprint is provided, so we need to read it out from the fingerprint
            self.sampling_distribution = self.__obtain_distribution_dict_from_fingerprint(
                path_to_fingerprint=sampler_config.class_sampling_dict)
            
        if sampler_config.background_as_random is None:
            self.background_as_random = 0.0
        else:
            self.background_as_random = sampler_config.background_as_random
                                  
    def __obtain_distribution_dict_from_fingerprint(
        self,
        path_to_fingerprint: str = None)->dict:
        """
        Method which obtains distribution for class sampling from the fignerprint file.
        It is simple, we take the fingerprint and based on the class statistcis stored in it (Median size), 
        for labels provided in self.labels we build up dictionary where key is label, and value is chance to
        sample that label. DISCLAIMER: all labels from the dataset must be provided!!
        
        Args:
            * path_to_fingerprint, str, path to the fingerprint file

        Returns:
            * dictionary with previously described key-value pairs for the labels obtained in self.labels
        """

        # Obtain fingerprint
        _fingerprint_dicts = read_data_fingerprint_json(
            path_to_json = path_to_fingerprint)

        # Obtain class statistics
        _class_statistics_dict = _fingerprint_dicts["CLASS_STATISTICS"]
        # Build dict with statitics
        _tmp_dict = {}

        # Obtain values
        for _label in self.labels:
            _tmp_dict[int(_label)] = _class_statistics_dict[str(int(_label))]["Median_size"]

        # Normalize
        # Calculate the sum of all values
        _total = sum(_tmp_dict.values())

        # Normalize the values
        _normalized_dict = {_key: _value / _total for _key, _value in _tmp_dict.items()}
        print(_normalized_dict)
        # Return
        return _normalized_dict

    def __iter__(self):
        # Custom logic to yield batches of indices. Logic describet in Init function
        while True:
            _batch = []
            # Obtain random samples for foreground
            if self.sampling_distribution is not None:
                # If distribution is given
                _classes = np.random.choice(
                    a = list(self.sampling_distribution.keys()),
                    size = self.number_of_fg_cases,
                    p = list(self.sampling_distribution.values()))
            else:
                # Uniform distribution
                _classes = np.random.choice(a = self.labels, size = self.number_of_fg_cases)
            
            # Obtain foregrounds
            for _i in range(self.number_of_fg_cases):
                _batch.append({"foreground": True,
                              "class": _classes[_i]
                              })            
            for _i in range(self.batch_size-self.number_of_fg_cases):
                _random_choice_for_background = np.random.choice([False, True], p=[self.background_as_random, 1 - self.background_as_random])
                
                if _random_choice_for_background:
                    # This is regular random - true random location
                    _batch.append({"foreground": False,
                                   "class": "-1"})
                else:
                    # If it is false we sample background
                    _batch.append({"foreground": False,
                                   "class": "-2"})
            random.shuffle(_batch)
            yield _batch


    def __len__(self):
        # Number of batches in the dataset
        return float('inf')
    
