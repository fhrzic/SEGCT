""" Implementation of low resolution simulation"""
from typing import Tuple
import torchio as tio
import torch
from torch.nn.functional import interpolate


class LowResolution(tio.Transform):
    """
    Class which implements random low resoltuion. How it does it. 
    Select_channel_of_iamge--> chose random scale per axis -->interpolate to that randomly chosen 
    dimension --> interpolate back to original dimension.
    """
    def __init__(self, 
                 scale_range: Tuple[float, float],
                 synchronize_channels: bool =False,
                 synchronize_axes: bool =True,
                 probability_per_channels: float = 0.5,
                 p: float = 1.0):
        """
        Init function of the low resolution transform. So how it works: p is a general probability 
        that this transfrom will occure. Let's say it is 0.5, it means there is 0.5 chance that this 
        probability will happen. Now, the low resolution is applied per each channel independently. 
        This means that although this augmentation is trigered and is applied to the input tensor, 
        it is possible that no transform occures because of the probability_per_channel is failed. 
        And yeah, scale range is U(float, float) to select. Depending on what is on, it is possible 
        to have sychronized (same scales) allong for all channels (synchronize channels) or along 
        all axes (synchornize_axes), or ofc both. 

        Args:
            * scale_range, Tuple[float, float], the scale will be sampled from this range for each 
            axis. 
            * synchronize_channels, bool, if true, all channels will have same scale.
            * synchronize,_axes, bool, if true, all axes inisde the same channel will have same 
            scale. 
            * probability_per_channel, float, probability that each channel get blured.
            * p, float, general probability that the whole bluring happens
        """
        # Inherit
        super().__init__(p = p)

        # Obtain scale ranges
        self.scale_range = scale_range

        # Syncrhonization
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes

        # Channel probability
        self.probability_per_channel = probability_per_channels

        # Upsample modes
        self.upmodes = {
            1: 'linear',
            2: 'bilinear',
            3: 'trilinear'
        }

    def apply_transform(self, subject: tio.Subject)->tio.Subject:
        """
        Method which applies transform. The transform is applied channel-wise. 

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

        # This was for testing purposes... you can't multiply integers and floats... idk
        if _image.dtype != torch.float32:
            _image = _image.to(torch.float32)


        # Generate sigmas (3 hours of my life to understand what the researchers
        # from nnUnet tried to do here)
        _scales = {}
        if self.synchronize_channels:
            # So all channels will have the same sigmas
            # Check for axes
            if self.synchronize_axes:
                # So all axes will have the same sigma
                _sigma_per_axis =  [torch.FloatTensor(1).uniform_(*self.scale_range).item()] \
                    * (len(_image.shape) - 1)
            else:
                # So all axes will have different sigmas
                _sigma_per_axis = [torch.FloatTensor(1).uniform_(*self.scale_range).item()
                                   for _ in range(len(_image.shape) - 1)]

            # Populate dict
            for _ch in range(_image.shape[0]):
                _scales[_ch] = _sigma_per_axis
        else:
            # So all channels will have different sigmas
            for _ch in range(_image.shape[0]):
                 # Check for axes
                if self.synchronize_axes:
                    # So all axes will have the same sigma
                    _sigma_per_axis =  [torch.FloatTensor(1).uniform_(*self.scale_range).item()] \
                        * (len(_image.shape) - 1)
                else:
                    # So all sigmas will have different axes
                    _sigma_per_axis = [torch.FloatTensor(1).uniform_(*self.scale_range).item() 
                                       for _ in range(len(_image.shape) - 1)]

                # Populate dict
                _scales[_ch] = _sigma_per_axis

        # For each channel generate probability to be applied
        _p_per_channels = torch.rand(_image.shape[0]) < self.probability_per_channel


        # Obtain original image shape (needed for reconstruction)
        _original_image_shape = _image.shape[1:]

        # Apply downsampling per channel
        for _ch, _apply_to_ch in enumerate(_p_per_channels):
            # If it is true, we apply transformation to channel
            if _apply_to_ch:
                # Calculate new shape by multiplying old shape with calculated scales
                _new_shape = [round(_orig_dim * _scale) 
                              for _orig_dim, _scale in zip(_original_image_shape, _scales[_ch])]

                # Downsample
                _downsampled_image = interpolate(
                    _image[_ch][None, None],
                    _new_shape,
                    mode='nearest-exact')

                # Upsample
                _image[_ch] = interpolate(
                    _downsampled_image,
                    _original_image_shape,
                    mode=self.upmodes[_image.ndim - 1])[0, 0]

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
            