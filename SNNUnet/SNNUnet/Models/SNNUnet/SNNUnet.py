""" Methods which builds SNNUnet"""
from typing import Tuple, List
import sys
import os
import torch
from torch import nn

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from Models.SNNUnet.building_blocks import *
from Models.SNNUnet.generate_SNNUnet_fingerprint import *

class Encoder(nn.Module):
    """
    Encoder modelue as described in NNUnet Paper
    """ 
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128,128,128),
        input_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        number_of_channels: int = 1,
        initial_features_number: int = 32,
        dropout: bool = False,
        feature_number_cap: int = 320,
        return_skip_layers: bool = True,
        z_last_channel: bool = False):

        """
        InitFunction for Encoder. 

        Args:
            * input_shape, Tuple(int,int,int), tuple which represents the input shape
            to the model
            * input_spacing, Tuple(float,float,float), tupple of floats which represents
            the spacing of each channel.
            * number_of_channels, integer, number of channels in the input image (required for first
            convolution)
            * initial_features_number, int, number of features in every convolutional block. After 
            each downsampling, the number of features is doubled (i.e. 32-->64-->128) until the 
            feature_number_cap is reached (i.e. feature number_sap = 320, 128-->256-->320-->320)
            * droput, bool, if set to True the dropout will be added in every convolution block with 
            the probability of 0.2. Please see buildingBlocks to see more about dropout. 
            * feature_number_cap, int, max number of features in conv block. See 
            initial_features_number for explanation.
            * return_skip_layers, bool, if set to True, the forward pass will return skip 
            connections outputs (outputs at the end of each conv block). This is added if someone 
            want to just use the encoder for i.e. classification, it can be just took and with this 
            flag used out of the box.
            * z_last_channel, bool, this bool says which axis is of plane one. If z_last_channel is 
            set to be true then (x,y,z) where z is out of the plane axis scenario applies. If 
            z_last_channel is set to false, then (z,x,y) scenario applies. This is important for 
            early stopping of nn building.

        Returns:
            * Nothing        
        """
        # Inherit
        super().__init__()

        # Set indexes for stopping constraint (special cases)
        if z_last_channel:
            _idx_x, _idx_y = (0,1)
        else:
            _idx_x, _idx_y = (1,2)

        # Skip layers
        self.skip_return = return_skip_layers

        # Make internal storage for outputs of every stage
        # which is necessary for skip connetions
        _block_storage = []
        self.downsampling_stride_storage = []
        self.kernel_shape_storage = []
        self.number_of_filters_storage = [initial_features_number]
        self.spacing = []
        self.shapes = []

        # Build blocks
        _spacing = input_spacing
        _shape = input_shape
        _number_of_in_channels = number_of_channels
        _number_of_filters = initial_features_number

        while not (_shape[0] // 2 <= 4 and _shape[1] // 2 <= 4 and _shape[2] // 2 <= 4) \
               or (_shape[_idx_x] == 8 and _shape[_idx_y] == 8 ):

            # Store spacing and shape
            self.spacing.append(_spacing)
            self.shapes.append(_shape)

            # Get block
            _block = ConvConvDownBlock(input_shape = _shape, 
                                      input_spacing = _spacing,
                                      number_of_input_channels = _number_of_in_channels,
                                      number_of_kernel_filters = _number_of_filters,
                                      dropout = dropout,
                                      feature_number_cap = feature_number_cap,
                                      downsample = True)

            # Obtain data for next iteration
            _number_of_filters, _shape, _spacing, _downsampling_stride, _kernel_size = \
                _block.get_output_shape()
            _number_of_in_channels = _number_of_filters
            self.number_of_filters_storage.append(_number_of_filters)

            # Store
            _block_storage.append(_block)
            self.downsampling_stride_storage.append(_downsampling_stride)
            self.kernel_shape_storage.append(_kernel_size)

            # Spacing check for anisotropy as described in nnUnet
            if max(_spacing) / min(_spacing) > 3.0:
                break

        # Add last block (bottleneck)
        # Get block
        _block = ConvConvDownBlock(input_shape = _shape, 
                                    input_spacing = _spacing,
                                    number_of_input_channels = _number_of_in_channels,
                                    number_of_kernel_filters = _number_of_filters,
                                    dropout = dropout,
                                    feature_number_cap = feature_number_cap,
                                    downsample = False)

        # Store
        _block_storage.append(_block)
        self.shapes.append(_shape)

        # Make it visible
        self.blocks = nn.Sequential(*_block_storage)

    def forward(
        self,
        x: torch.Tensor = None)->List[torch.Tensor]:
        """
        Forward pass of the Encoder

        Args: 
            * x, torch.Tensor, input torch tensor

        Returns:
            * if return_skip_layers is true, this encoder will output list of the 
            torch tensors y_i, where every y represents one of the i skip connections. 
            If return_skip_layers is false, this module will return torch.Tensor (NOT LIST)
            of the output
        """
        # Storage for returns
        _return_values = []

        # Obtain predictions
        for _block in self.blocks:
            x, _aux = _block(x)
            _return_values.append(_aux)

        # Return all or just the last one
        if self.skip_return:
            # LIST
            return _return_values
        else:
            # TENSOR
            return _return_values[-1]

    def obtain_fingerprint(
        self)->List[list]:
        """
        Method which returns fingerprint information such as stride and conv kernels.
        This is necessary for the decoder part of the nn.
        
        Returns
            * list of strides
            * list of conv kernels
            * list of features
            * list of shapes
            * list of spacings
        """

        return [self.downsampling_stride_storage, 
                self.kernel_shape_storage, 
                self.number_of_filters_storage,
                self.shapes,
                self.spacing]

class Decoder(nn.Module):
    """
    Decoder part of the Unet created as explained in NNUnet paper
    """

    def __init__(
        self,
        upsampling_stride_list : List[Tuple[int,int,int]] = None,
        conv_kernels_list : List[Tuple[int,int,int]] = None,
        features_list : List[int] = None,
        output_auxiliary_predictions: bool = True,
        number_of_output_classes: int = 7):
        """
        Init function of the decoder containing all necessary building blocks of the decoder

        Args:
            * upsampling_stride_list, List of Tupples(int,int,int), this list is obtained by encoder
            and represents the reverse order of strides necessary for upsampling of the kernels.
            * conv_kernel_list, List of Tupples(int,int,int), this list is obtained by encoder and 
            represents the reverse number of kernel shapes used in convolutions (not all conv 
            kernels are 3x3x3) due to the spacing).
            * features_list, List of integers, number of features in every block. It is in reverse 
            oreder from the encoder (e.g. 320-->256-->128). It is easier to get it from encoder 
            than to recaluclate which involves spacing again.
            * output_auxiliary_predictions, bool, if set to true after each block small "head"
            called miniblock is added which which chenges the output to the desired number of 
            outputclasses and performs softmax
            * number_of_output_classes, int, number of output clsses which means number of 
            output channels.
        """
        # Inherit
        super().__init__()

        # Set auxiliary predictions global
        self.output_auxiliary_predictions = output_auxiliary_predictions

        # Storage for blocks
        _block_storage = []

        # Build up the blocks
        for _i, _stride in enumerate(upsampling_stride_list):
            # Obtain upsampling block
            _block = UpConvConvBlock(number_of_input_channels = features_list[_i],
                                     number_of_output_channels = features_list[_i+1],
                                     upsample_stride = _stride, 
                                     kernel_size = conv_kernels_list[_i],
                                     dropout = False,
                                     number_of_output_classes = number_of_output_classes,
                                     return_auxiliary_output = output_auxiliary_predictions)           

            # Add it to all blocks
            _block_storage.append(_block)

        # Make block available
        self.blocks = nn.Sequential(*_block_storage)

        # In case that the auxilary predictions is set to non, we still need to perform
        # the last one (final output)
        # This is just 1x1x1 conv wiht softmax
        self.output_miniblock = OutputMiniBlock(
            number_of_input_channels = features_list[-1],
            number_of_output_channels = number_of_output_classes)

    def forward(
        self,
        skip_connections: List[torch.Tensor] = None)->List[torch.Tensor]:
        """
        Forward method for the decoder.

        Args:
            * skip_connections, list of torch.Tensor which are the skip connections from the encoder
        """
        # Storage for returns
        _return_values = []

        # First one ic the beggining
        _x = skip_connections[0]

        # Apply collected blocks
        for _block, _connection in zip(self.blocks, skip_connections[1:]):
            _x, _aux_output = _block(_x, _connection)

            # Get miniblock (output)
            _return_values.append(_aux_output)

       # Return all or just the last one
        if self.output_auxiliary_predictions:
            # LIST
            return _return_values
        else:
            # TENSOR
            # Add outputMiniBlock
            _x = self.output_miniblock(_x)
            return _x

class SNNUnet(nn.Module):
    """
    Main class for building Simple No-New Unet
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = None,
        input_spacing: Tuple[float, float, float] = None,
        number_of_channels: int = 1,
        z_last_channel: bool = False,
        number_of_output_classes: int = 7,
        output_auxiliary_predictions: bool = True):
        """
        Init for building SNNunet. There are much more paramteres which can be specified,
        but this are the minumum required to build the NNUnet. All other parameters are 
        set to be as the one in NNUnet. Nothing stops user to change them.

        Args:
            * input_shape, tuple(int, int, int), input size of a volume.
            * input_spacing, tuple(float, float, float), spacing of the volume. 
            * number_of_channels, int, number of channels of the input volume.
            * z_last_channel, bool,  WE ASUME HERE THAT Z CHANNEL IS FIRST: ((z), x, y). If
            set to True: (x, y, z). Z is the axis which is not in-plane.
            * number_of_output_classes, int, number of output classes.
            * output_auxiliary_predictions, bool, if true, returns output of each decoder block.
              During the estimation of batch size it is alwys kept true (number of params added 
              is negligable)
        """

        # Inherit
        super().__init__()

        # Define encoder
        self.encoder = Encoder(input_shape = input_shape,
                               input_spacing = input_spacing,
                               number_of_channels = number_of_channels,
                               z_last_channel = z_last_channel)

        # Obtain decoders fingerprint
        _dec_downsampling_strides, _dec_krenel_shapes, \
        _dec_number_of_filters, _dec_shapes, _dec_spaces = self.encoder.obtain_fingerprint()

        # Reverse the ordering
        _dec_downsampling_strides.reverse()
        _dec_krenel_shapes.reverse()
        _dec_number_of_filters.reverse()
        self.shapes = _dec_shapes
        self.spaces = _dec_spaces

        # Obtain decoder
        self.decoder = Decoder(upsampling_stride_list = _dec_downsampling_strides,
                                conv_kernels_list = _dec_krenel_shapes,
                                features_list = _dec_number_of_filters,
                                output_auxiliary_predictions = output_auxiliary_predictions,
                                number_of_output_classes = number_of_output_classes)

    def forward(
        self,
        x: torch.Tensor = None)->List[torch.Tensor]:
        """
        Forward function of the SSNunet

        Args:
            * x, torch.Tensor, input to the NNUnet
        
        Returns:
            * list of torch.Tensors (auxiliary predictions) in order from top one to bottom 
            (last calculated to first caluclated)
        """
        # Calculate encoder
        _x = self.encoder(x)

        # Reverse order so it is easier to handle decoder
        _x.reverse()

        # Apply decoder
        _x = self.decoder(_x)

        # Return
        return _x

    def get_output_shapes(self)->List[List[Tuple[int, int, int]]]:
        """
        Method which returns dict containing shapes and spaces. Shapes and spaces are sorted 
        from the first layer of the decoder to the last one (from last output to the bottlneck, 
        full resolution to the lowest resolution). The encoder returns them in the oposite direction.

        Returns:
            * two lists where first list represents output shapes and the last list represent output
            spaces (in order from the last outputed to the first outputed)
        """

        # Return
        return [self.shapes, self.spaces]
 