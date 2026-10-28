""" 
Script which implements deep supervision resize.
TorchIO have Resize as augmentation, but I am not feeling it 
to use it. It has warnings etc.
"""

from typing import Tuple, List
import torchio as tio
import torch
from torch.nn.functional import interpolate

class DeepSupervisionResize(tio.Transform):
    """
    Class which implements random brigthness transform per channel of the image
    """
    def __init__(
        self,
        shapes_list: List[List[int]] = None,
        p: float = 1):
        """
        Init function for contrast transform.

        Args:
            * shapes_list, List[List(int,int,int)], this is for deep supervision so it 
            expects list of shapes to output. Each shape is (int, int, int). This
            list can be obtained from model_fingerprint.json (if the SNNUnet is used)
            * p, float, probability to apply this transform, this should be set to 1 because
            we always want to apply it in case of deep supervision
        """

         # Inherit
        super().__init__(p = p)

        # Obtain contrast ranges
        self.output_shapes = shapes_list

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Method which applies transfrom and caluclates all necessery stuff
        Args:
            * subject, tio.Subject, subject to which the  transformation will be applied

        Returns:
            * transformed subject
        """

        # Get image as tensor IMPORTANT: THIS IS DONE ON THE LABEL --> segmentation mask
        # The resized outputs for Image will be obtained from the model.
        if isinstance(subject, tio.Subject):
            _label = subject['label'].data
        else:
            if isinstance(subject, tio.ScalarImage):
                _label = subject.data
            else:
                _label = subject

        # Because interpolation in torch functional expect 5D input, we need to unsqueeze
        # Create storage
        _labels = []
        # The input label 3D-->5D
        _label = torch.unsqueeze(_label, 0)

        # Resize the input and
        for _targeted_shape in self.output_shapes:
            # Resize
            _resized_label = interpolate(
                input =_label.float(),
                size = _targeted_shape,
                mode='nearest-exact')
            # Remove added channels and batch sizes
            _resized_label = torch.squeeze(_resized_label, 0)

            # Add it to the output
            _labels.append(_resized_label)

        # Revert the order of the labels because the model returns predictions from
        # bottleneck to the output
        _labels.reverse()

        # Build back subject and return
        if isinstance(subject, tio.Subject):
            subject['label'].data = torch.squeeze(_label, 0) # Original
            subject['ds_labels'] = _labels
            return subject
        else:
            if isinstance(subject, tio.ScalarImage):
                subject.data = _labels
                return subject
            else:
                return _labels
