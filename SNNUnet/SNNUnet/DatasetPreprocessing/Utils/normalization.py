""" Normalization for the volumes and segmentations"""

import os
import sys
import numpy as np

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from DatasetFingerprint.Utils.utility_functions import read_data_fingerprint_json

class NormalizeCTFromFingerprint():
    """
    Class which normalize ct. Same as nnUnet
    """

    def __init__(self, path_to_fingerprint: str = None)->np.ndarray:
        """
        Args:
            * path_to_fingerprint, str, path to the json containing fingerprint data generated
            by FingerprintGenerationScript.py in DataFingerprint      
        """
        # Read data and obtrain json
        self.properties = {}
        try:
            _dictionaries = read_data_fingerprint_json(path_to_fingerprint)
            self.properties = _dictionaries["VOXEL_STATISTICS_SAMPLES"]
        except ValueError as _exc:
            raise ValueError(
                f"Something went wrong in CT Normalization." 
                f"Could not obtain data from {path_to_fingerprint}!") from _exc

    def run(self, 
            volume: np.ndarray = None)->np.ndarray:
        """
        Main method to be called for running normalization. It normalize CT same as it is done in
        nnUnet. Normalization is done in two steps:

        1.) Data is clipped to the min/max value provided in fingerprint.json
        2.) Data is normalize with Z normalization (mean&std) 

        Args:
            * volume, np.ndarray, volume representing ct which is going to be normalized  

        Returns:
            * np.ndarray -> normalized version of numpy array based on the provided json file
        """
        # Obtain parameters
        mean_intensity = self.properties['mean']
        std_intensity = self.properties['std']
        lower_bound = self.properties['percentile_00_5']
        upper_bound = self.properties['percentile_99_5']

        # Clip
        _volume = np.zeros(volume.shape, dtype=np.float32)
        np.clip(volume, lower_bound, upper_bound, out=_volume)

        # Z normalize
        _volume -= mean_intensity
        _volume /= max(std_intensity, 1e-8)

        # Return
        return _volume

class NormalizeSegmentationsFromFingerprint():
    """
    DISCLAIMER: ONLY FOR OUR DATASET (NOT NECESSARY FOR OTHERS)

    Class which normalize labels. Namely, due to the export, some segmentation maps
    may contain some lables which are not relelvant. Hence, we decided to fix them
    by setting all labels which are erroneous to background
    """

    def __init__(self, path_to_fingerprint: str = None):
        """
        Args:
            * path_to_fingerprint, str, path to the json containing fingerprint data generated
            by FingerprintGenerationScript.py in DataFingerprint      
        """
        # Read data and obtain json
        try:
            _dictionaries = read_data_fingerprint_json(path_to_fingerprint)
            _labels = _dictionaries["LABELS_MAPPING"]
            self.white_list = list(_labels.keys())
            self.white_list = [float(_item) for _item in self.white_list]
        except ValueError as _exc:
            raise ValueError(
                f"Something went wrong in Segmentation Normalization." 
                f"Could not obtain data from {path_to_fingerprint}!") from _exc

    def run(self, 
            volume: np.ndarray = None)->np.ndarray:
        """
        Main method to be called for running lable checking. 

        Args:
            * volume, np.ndarray, volume representing segmentation array 
            which is going to be check for correct labels

        Returns:
            * np.ndarray -> normalized version of numpy array based on the provided json file
        """
        # Obtain invalid indices
        _mask = ~np.isin(volume, self.white_list)

        # Set all values to 0 in mask
        volume[_mask] = 0

        # Return
        return volume

class NormalizeCTFromVolume():
    """
    Class which normalize ct. This is normalization of every case on its own. Every
    case is normalized for itself based on the data statistics found in that case.
    """

    def __init__(self):
        """
        Everything is calculated based on the provided volume      
        """

    def run(self,
            volume: np.ndarray = None)->np.array:
        """
        It normalize CT on case basis. Simply every case is normalized based on its own statistics
        by utilizing z-nomralization method.

        Args:
            * volume, np.ndarray, volume representing ct which is going to be normalized  

        Returns:
            * np.ndarray -> normalized version of numpy array based on the provided json file
        """
        # In case of inifinites, we set them to 0

        # Clip the values to be within the range of float32
        _volume = np.clip(volume, -1e-8, 1e8)

        # Set type
        _volume = _volume.astype(np.float32, copy = False)        
         
        # Just a simple check because we are not clipping values
        if not np.all(np.isfinite(_volume)):
            print("Warning: volume contains non-finite values (e.g., -inf, +inf, or NaN)")
        
        # Calculate params
        _mean_intensity = _volume.mean()
        _std_intensity = _volume.std()
        
        # Z normalize
        _volume -= _mean_intensity        
        _volume /= max(_std_intensity, 1e-8)

        # Return
        return _volume
    
class NormalizeCTFromMask():
    """
    This normalization is similar to the NormalizeCT, but instead of taking all voxels into account
    when calculating mean and std, only those where Segmentation Mask is having a value > 0 are 
    contributing to the caluculation
    """

    def __init__(self):
        """
        Everything is calculated based on the provided volume      
        """

    def run(self,
            volume: np.ndarray = None,
            segmentation: np.ndarray = None)->np.array:
        """
        It normalize volume based on the values of the fg voxels presnet in segmentation.

        Args:
            * volume, np.ndarray, volume representing ct which is going to be normalized  
            * segmentation, np.ndarray, volume representing segmentation mask 

        Returns:
            * np.ndarray -> normalized version of numpy array based on the fg pixels
        """
        # Clip the values to be within the range of float32
        _volume = np.clip(volume, -1e-8, 1e8)

        # Set type
        _volume = _volume.astype(np.float32, copy = False)              

        # Just a simple check because we are not clipping values
        if not np.all(np.isfinite(_volume)):
            print("Warning: _volume contains non-finite values (e.g., -inf, +inf, or NaN)")
            

        # Calculate mask based on the segmentation
        _mask = segmentation > 0
        _mean_intensity = volume[_mask].mean()
        _std_intensity = volume[_mask].std()

        # Z normalize
        _volume -= _mean_intensity
        _volume /= max(_std_intensity, 1e-8)

        # Return
        return _volume
    