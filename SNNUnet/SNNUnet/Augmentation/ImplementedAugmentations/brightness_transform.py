""" Implementation of brightness transform like in NNUnet"""
from typing import Tuple
import torch
import torchio as tio

class BrightnessTransform(tio.Transform):
    """
    Class which implements random brigthness transform per channel of the image
    """
    def __init__(
        self,
        intensity_range: Tuple[float, float],
        synchronize_channels: bool = False,
        probability_per_channels: float = 1.0,
        p: float = 1.0):
        """
        Init function for brigness transform.

        Args:
            * intensity range, [float, float], random multiplier of brightness from the range
            * synchronize_channels, bool, if true, all channels are multiplied with same insity 
            chosen from intensity range, otherwise, for each channel separate multiplier is chosen.
            * probability_per_channel, float, each channel has this probability to have brightness 
            adjustment.   
            * p, float, probability to apply this transform
        """
        # Inherit
        super().__init__(p = p)

        # Obtain intesity renges
        self.intensity_range = intensity_range

        # Syncrhonization
        self.synchronize_channels = synchronize_channels

        # Channel probability
        self.probability_per_channel = probability_per_channels

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Method which applies transfrom and caluclates all necessery stuff
        Args:
            * subject, tio.Subject, subject to which the  transformation will be applied

        Returns:
            * transformed subject
        """

        # Get image as tensor
        if isinstance(subject, tio.Subject):
            _image = subject['image'].data
        else:
            if isinstance(subject, tio.ScalarImage):
                _image = subject.data
            else:
                _image = subject

        # Generate multipliers per channels
        if self.synchronize_channels:
            # All channels get same multiplier
            _multipliers =  [self.__sample_brightness(intensity_range = self.intensity_range)] \
                * (_image.shape[0])
        else:
            # Different multiplier for each channel
            _multipliers =  [self.__sample_brightness(intensity_range = self.intensity_range)
                             for _ in range(_image.shape[0])]
        # For each channel generate probability to be applied
        _p_per_channels = torch.rand(_image.shape[0]) <= self.probability_per_channel

        # This was for testing purposes... you can't multiply integers and floats... idk
        if _image.dtype != torch.float32:
            _image = _image.to(torch.float32)

        # Apply
        for _ch in range(_image.shape[0]):
            # Apply if it should be applied
            if _p_per_channels[_ch]:
                _image[_ch] *= _multipliers[_ch]

        # Build back subject and return
        if isinstance(subject, tio.Subject):
            subject['image'].data = _image
            return subject
        else:
            if isinstance(subject, tio.ScalarImage):
                subject.data = _image
                return subject
            else:
                return _image

    def __sample_brightness(self, intensity_range: Tuple[float, float])->float:
        """
        This method helps sampling the intesnity value from the given intesnity_range 
        [min-float, max-float].
        It applies policy:
            --> with p = 0.5 work on low boundary
                --> if low boundary, selects sample intesnity from [min-float, 1] 
                (reducing the intesnity) 
                    Only in case min-float is < 1
            else
                --> high boundars, sample from [max(min-float, 1), max-float] 
                (increasing the intesnity)
                Always implemented if min-float is < 1

        Args:
            * intenstiy_range, Tuple[float, float], where first float is min-float 
            (minimum of the range), 
            and max-float is maximum of the range
        
        Returns:
            * float, number which is randomly sampled by described policies
        """
        # Obtain random number between 0 and 1
        _random_probability = torch.rand(1).item()

        # Sample random probability
        if _random_probability < 0.5 and intensity_range[0] < 1:
            _factor = torch.FloatTensor(1).uniform_(intensity_range[0], 1)
        else:
            _factor = torch.FloatTensor(1).uniform_(max(intensity_range[0], 1), intensity_range[1])
        # Return
        return _factor
    