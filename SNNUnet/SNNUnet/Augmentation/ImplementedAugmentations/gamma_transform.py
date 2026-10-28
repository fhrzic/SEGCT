""" Gamma transform same as in NN-Unet"""
from typing import Tuple
import torchio as tio
import torch


class GammaTransform(tio.Transform):
    """
    Class which implements random gamma transform per channel of the image
    """
    def __init__(self, 
                gamma_scale: Tuple[float, float],
                invert_image: bool = True,
                synchronize_channels: bool = False,
                probability_per_channels: float = 1,
                retain_stats: bool = True,
                p: float = 0.1):
        """
        Init function for contrast transform.

        Args:
            * gamma_scale, float, float, random multiplier of brightness from the range
            * invert_image, bool, inverting image colour
            * synchronize_channels, bool, if true, all channels are multiplied with same gamma 
            value sampled from gamma_scale.
            * probability_per_channel, float, each channel have this probability to be affected 
            by this augmentation method
            * retain_stats, bool, if True, the mean and std will be preserved when transfered 
            to new "gamma multiplied" space. 
            * p, float, probability to apply this transform   
        """

         # Inherit
        super().__init__(p = p)

        # Obtain gamma ranges
        self.gamma_scale = gamma_scale

        # Invert Image (True or false)
        self.invert_image = invert_image

        # Synchroniz channels - all channels have the same gamma
        self.synchronize_channels = synchronize_channels

        # Channel probability
        self.probability_per_channel = probability_per_channels

        # Retain stats
        self.retain_stats = retain_stats

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
            _multipliers =  [self.__sample_gamma(gamma_range = self.gamma_scale)] \
                * (_image.shape[0])
        else:
            # Different sigma for each channel
            _multipliers =  [self.__sample_gamma(gamma_range = self.gamma_scale)
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

                # First invert image if necessary
                if self.invert_image:
                    _image *= -1

                # Memorize the original stats
                if self.retain_stats:
                    _mean_org = torch.mean(_image[_ch])
                    _std_org = torch.std(_image[_ch])           

                # Apply gamma
                _minm = torch.min(_image[_ch])
                _rnge = torch.max(_image[_ch]) - _minm
                _image[_ch] = torch.pow(((_image[_ch] - _minm) / torch.clamp(_rnge, min=1e-7)),
                                        _multipliers[_ch]) * _rnge + _minm

                # If we need to retain values, here is where we apply original min and std
                if self.retain_stats:
                    _mean_new = torch.mean(_image[_ch])
                    _std_new = torch.std(_image[_ch])
                    _image[_ch] -= _mean_new
                    _image[_ch] *= (_std_org / torch.clamp(_std_new, min=1e-7))
                    _image[_ch] += _mean_org

                # Return image inverse
                if self.invert_image:
                    _image[_ch] *= -1

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

    def __sample_gamma(self, gamma_range: Tuple[float, float])->float:
        """
        This method helps sampling the gamma value from the given gamma_range 
        [min-float, max-float].
        It applies policy:
            --> with p = 0.5 work on low boundary
                --> if low boundary selects sample gamma from [min-float, 1] (reducing the gamma) 
                    Only in case min-float is < 1
            else
                --> high boundars, sample from [max(min-float, 1), max-float] (increasing the gamma)
                Always implemented if min-float is < 1

        Args:
            * gamma_range, Tuple[float, float], where first float is min-float 
            (minimum of the range), and max-float is maximum of the range
        
        Returns:
            * float, number which is randomly sampled by described policies
        """
        # Obtain random number between 0 and 1
        _random_probability = torch.rand(1).item()

        # Sample random probability
        if _random_probability < 0.5 and gamma_range[0] < 1:
            _factor = torch.FloatTensor(1).uniform_(gamma_range[0], 1)
        else:
            _factor = torch.FloatTensor(1).uniform_(max(gamma_range[0], 1), gamma_range[1])
        # Return
        return _factor
    