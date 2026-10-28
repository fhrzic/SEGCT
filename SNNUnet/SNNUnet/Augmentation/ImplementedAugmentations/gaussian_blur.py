"""Implementation of Gaussian Blur the same way it is implemented in NNUnet"""
from typing import Tuple
import torchio as tio
import torch
import torch.nn.functional as F

class GaussianBlur(tio.Transform):
    """
    Class which implements random gaussian blur per each axis same as proposed in nnUnet 
    gaussian_blur.py. However, this works with torchio
    """
    def __init__(
        self, 
        sigma_range: Tuple[float, float],
        probability_per_channel: float = 1.0,
        synchronize_channels: bool = False,
        synchronize_axes: bool = False,
        p: float = 1.0):
        """
        Init function of the transform. So how it works: p is a general probability that 
        this transfrom will occure. Let's say it is 0.5, it means there is 0.5 chance that 
        this probability will happen. Now, the gaussian blur is applied per each channel
        independently with chance probabilit_per_channel. This means that although this 
        augmentation is trigered and is applied to the input tensor, it is possible that 
        no transform occures because of the probability_per_axis is failed. And yeah, 
        sigma range is U(float, float) to apply noise. Depending on what is on, it is 
        possible to have sychronized (same sigmas) allong for all channels 
        (synchronize channels) or along all axes (synchornize_axes)

        Args:
            * sigma_range, Tuple[float, float], the sigma will be sampled from this 
            range for each axis. Bigger sigma more spread out kernel, smaller signma 
            will result in narrower kernel.
            * probability_per_channel, float, probability that each channel get blured.
            * synchronize_channels, bool, if true, all channels will have same sigma.
            * synchronize,_axes, bool, if true, all axes inisde the same channel will 
            have same sigma.
            * p, float, general probability that the whole bluring happens
        """
        # Inherit
        super().__init__(p = p)

        # Obtain sigma
        self.sigma_range = sigma_range

        # Syncrhonization
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes

        # Channel probability
        self.probability_per_channel = probability_per_channel

    def apply_transform(self, subject: tio.Subject) ->tio.Subject:
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

        # Generate sigmas (3 hours of my life to understand what the researchers from nnUnet
        # tried to do here)
        _sigmas = {}
        if self.synchronize_channels:
            # So all channels will have the same sigmas
            # Check for axes
            if self.synchronize_axes:
                # So all axes will have the same sigma
                _sigma_per_axis =  [torch.FloatTensor(1).uniform_(*self.sigma_range).item()] \
                    * (len(_image.shape) - 1)
            else:
                # So all axes will have different sigmas
                _sigma_per_axis = [torch.FloatTensor(1).uniform_(*self.sigma_range).item()
                                   for _ in range(len(_image.shape) - 1)]

            # Populate dict
            for _ch in range(_image.shape[0]):
                _sigmas[_ch] = _sigma_per_axis
        else:
            # So all channels will have different sigmas
            for _ch in range(_image.shape[0]):
                 # Check for axes
                if self.synchronize_axes:
                    # So all axes will have the same sigma
                    _sigma_per_axis =  [torch.FloatTensor(1).uniform_(*self.sigma_range).item()] \
                        * (len(_image.shape) - 1)
                else:
                    # So all sigmas will have different axes
                    _sigma_per_axis = [torch.FloatTensor(1).uniform_(*self.sigma_range).item()
                                       for _ in range(len(_image.shape) - 1)]

                # Populate dict
                _sigmas[_ch] = _sigma_per_axis

        # For each channel generate probability to be applied
        _p_per_channels = torch.rand(_image.shape[0]) < self.probability_per_channel

        # Apply gausian to each channel
        for _ch, _apply_to_ch in enumerate(_p_per_channels):
            # This channel is skipped due to the probability
            if _apply_to_ch == False:
                continue

            # Ok finaly, we are applying blur
            _image[_ch] = self.__apply_gaussian_blur_3d(img = _image[_ch], 
                                                        sigmas = _sigmas[_ch])

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


    def __apply_gaussian_blur_3d(
        self,
        img: torch.Tensor,
        sigmas: Tuple[float, float, float]) -> torch.Tensor:
        """
        Apply Gaussian blur separately along each axis (x, y, z) using separable 1D 
        convolutions. This is applied on each chanel.

        Args:
            * img, torch.Tensor, input image which will be transformed.
            * sigmas, Tuple, three floats representing sigmas for each axis 
        
        Returns:
            * image which is blured along axes    
        """
        # Obtain sigmas
        _sigma_x, _sigma_y, _sigma_z = sigmas

        # Build a unique kernel for each axis
        _kernel_x = self.__build_kernel(sigma = _sigma_x).view(1, 1, -1, 1, 1)  # x-axis
        _kernel_y = self.__build_kernel(sigma = _sigma_y).view(1, 1, 1, -1, 1)  # y-axis
        _kernel_z = self.__build_kernel(sigma = _sigma_z).view(1, 1, 1, 1, -1)  # z-axis

        _img_channel = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # This was for testing purposes... you can't multiply integers and floats... idk
        if _img_channel.dtype != torch.float32:
            _img_channel = _img_channel.to(torch.float32)

        # Apply Gaussian blur along each axis independently
        _blurred = F.conv3d(_img_channel, _kernel_x, padding=(_kernel_x.shape[2] // 2, 0, 0))
        _blurred = F.conv3d(_blurred, _kernel_y, padding=(0, _kernel_y.shape[3] // 2, 0))
        _blurred = F.conv3d(_blurred, _kernel_z, padding=(0, 0, _kernel_z.shape[4] // 2))

        # Return blured channel along all axes but we need to unsqueze it.
        return _blurred.squeeze(0).squeeze(0)  # Remove added batch and channel dims

    def __build_kernel(
        self, 
        sigma: float,
        truncate: float = 6) -> torch.Tensor:
        """
        Build a 1D Gaussian kernel based on the given sigma and truncate value.

        Args:
            * sigma, float, The standard deviation of the Gaussian distribution.
            * truncate, float, Truncate the kernel at this many standard deviations.

        Returns:
            * torch.Tensor, A 1D Gaussian kernel tensor.
        """
        # Compute the kernel size based on sigma and truncate
        _kernel_size = self.__compute_kernel_size(sigma, truncate)
        _half_size = (_kernel_size - 1) * 0.5

        # Generate the range of values centered at zero
        _x = torch.linspace(-_half_size, _half_size, steps = _kernel_size)

        # Apply the Gaussian function
        _gaussian_pdf = torch.exp(-0.5 * (_x / sigma).pow(2))

        # Normalize the kernel so that the sum is 1
        _kernel_1d = _gaussian_pdf / _gaussian_pdf.sum()

        return _kernel_1d

    def __round_to_nearest_odd(self, value: float) -> int:
        """
        Round a value to the nearest odd integer.

        Args:
            * value, float, the input value to round.

        Returns:
            * int, the nearest odd integer.
        """
        rounded_value = round(value)

        # If already odd, return the rounded value
        if rounded_value % 2 == 1:
            return rounded_value

        # If even, adjust to the nearest odd number
        return rounded_value + 1 if value - rounded_value >= 0 else rounded_value - 1

    def __compute_kernel_size(self, sigma: float, truncate: float = 4) -> int:
        """
        Compute the size of the Gaussian kernel based on sigma and truncate.

        Args:
            * sigma, float, The standard deviation of the Gaussian distribution.
            * truncate, float, Truncate the kernel at this many standard deviations.

        Returns:
            * int: The size of the Gaussian kernel, rounded to the nearest odd integer.
        """
        # Calculate kernel size based on sigma and truncate, and round to the nearest odd integer
        _kernel_size = self.__round_to_nearest_odd(sigma * truncate + 0.5)
        # Return
        return _kernel_size
    