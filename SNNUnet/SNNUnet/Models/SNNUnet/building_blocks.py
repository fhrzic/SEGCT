""" Building blocks for SNNUnet"""
from typing import Tuple, List
import torch
from torch import nn

class ConvConvDownBlock(nn.Module):
    """
    This is main encoder block. The block is build as follows:
    
    1.) 3D Conv 
        1.1) Dropout (p=02)
    2.) Insctance normalisation 3D
    3.) Leaky Relu  (negative slope 0.01)

    4.) 3D Conv 
        4.1) Dropout(p=0.2)
    5.) Insctance normalisation 3D
    6.) Leaky Relu  (negative slope 0.01)

    7.) Downslampling = Strided Convolution
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128,128,128),
        input_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        number_of_input_channels: int = 3,
        number_of_kernel_filters: int = 32,
        dropout: bool = False,
        feature_number_cap: int = 320,
        downsample: bool = True):
        """
        Init method of the module. Main building block of encoder.

        Args:
            * input_shape, Tuple(int, int, int), input shape to the conv block.
            * input_spacing, Tuple(float, float, float), input spacing to the conv block. Needed to 
            estimate kernel dimensions and downsampling strides
            * number_of_input_channels, int, number of input channels to the kernel.
            * number_of_kernel_filters, int, number of kernel filters (filters in conv layer)
            * dropout, bool, if set to True, dropout with p = 0.2 will be applied to the both 
            convolutions in the block.
            * feature_num_cap, int, max number of features (kernel_filters) in conv layers. 
            * downsample, bool, if set to false the downsampling will not be performed 
            (usefull for bottleneck)
        """
        # Super
        super(ConvConvDownBlock, self).__init__()

        # Set downsampling
        self.downsample = downsample

        # Obtain stride and kernel for first convolution
        ## Max value
        _min_value = min(input_spacing)

        ## Now generate stride by applaying the following rule:
        ## if the value is double the size than the min value, it is intact
        ## otherwise it is halved
        _downsampling_stride = tuple(1 if _x > 2 * _min_value else 2 for _x in input_spacing)

        # Check if downsampling would result in under 4, if that is the case, set stride to 1
        _downsampling_stride = tuple(1 if _x // 2  < 4 else _y for _x, _y in \
                                     zip(input_shape, _downsampling_stride))
        self.output_spacing = [_x*_y for _x, _y in zip(input_spacing, _downsampling_stride)]
        self.output_shape = [_x // _y for _x, _y in zip(input_shape, _downsampling_stride)]
        self.downsampling_stride = _downsampling_stride

        # Calculate convs kernel
        ## Same logic as for downsampling stride except we use 3x3x3
        _kernel_size = tuple(1 if _x > 2 * _min_value else 3 for _x in input_spacing)
        self.kernel_size = _kernel_size

        ## Kernel stride is always 1x1x1 (same)
        _kernel_stride = (1,1,1)

        # Set dropout flag
        self.apply_dropout = dropout

        # Build layers
        self.dropout_1 = nn.Dropout3d(p = 0.2)
        self.dropout_2 = nn.Dropout3d(p = 0.2)

        # First convolution
        self.conv_1 = nn.Conv3d(in_channels = number_of_input_channels,
                                    out_channels = number_of_kernel_filters,
                                    kernel_size = _kernel_size,
                                    stride = _kernel_stride,
                                    padding = [(_i - 1) // 2 for _i in _kernel_size],
                                    dilation = 1,
                                    bias = True)
        self.conv_2 = nn.Conv3d(in_channels = number_of_kernel_filters,
                                out_channels = number_of_kernel_filters, 
                                kernel_size = _kernel_size, 
                                stride = _kernel_stride,
                                padding = [(_i - 1) // 2 for _i in _kernel_size],
                                dilation = 1, 
                                bias = True)

        # Instance normalization
        self.instance_norm_1 = nn.InstanceNorm3d(num_features = number_of_kernel_filters,
                                                 eps = 1e-05,
                                                 affine = True)
        self.instance_norm_2 = nn.InstanceNorm3d(num_features = number_of_kernel_filters,
                                                 eps = 1e-05,
                                                 affine = True)

        # Activation function
        self.l_relu_1 = nn.LeakyReLU(negative_slope = 1e-2)
        self.l_relu_2 = nn.LeakyReLU(negative_slope = 1e-2)

        # Downsampling == strided convolution
        self.output_number_of_channels = min(number_of_kernel_filters * 2, feature_number_cap)
        self.strided_convolution = nn.Conv3d(in_channels = number_of_kernel_filters,
                                             out_channels = self.output_number_of_channels, 
                                             kernel_size = _kernel_size, 
                                             stride = _downsampling_stride, 
                                             padding = [(_i - 1) // 2 for _i in _kernel_size], 
                                             dilation = 1, 
                                             bias = True)

    def forward(
        self,
        x:torch.Tensor)->List[torch.Tensor]:
        """
        Forward method which applies all elements of a block

        Args:
            * x, torch.Tensor, input Tensor of a block
        
        Returns:
            * 2 tensors: torch.Tensor, tensor after the operations are made over the X

        """
        # Mini block 1
        _x = self.conv_1(x)
        if self.apply_dropout:
            _x = self.dropout_1(_x)
        _x = self.instance_norm_1(_x)
        _x = self.l_relu_1(_x)

        # Mini block 2
        _x = self.conv_2(_x)
        if self.apply_dropout:
            _x = self.dropout_2(_x)
        _x = self.instance_norm_2(_x)
        _auxiliary_output = self.l_relu_2(_x)

        # Downsampling
        if self.downsample:
            _x = self.strided_convolution(_auxiliary_output)
        else:
            _x = _auxiliary_output
        # Return
        return _x, _auxiliary_output

    def get_output_shape(self)->Tuple[int, tuple, tuple, tuple, tuple]:
        """
        Method which returns shapes of the outputs (after performing blocks).
        
        Returns:
            * output_number_of_channels, int, output number of features after downsampling 
            * output_shape, (int, int ,int), shape of the output after performing defined blocks
            * output_spacing, (float, float, float), output spacing after performing defined blocks
            * downsampling_stride, (int, int, int), stride in performed downsampling
            * kernel_size, (int, int int), used kernel size in convolutions
        """
        # Return
        return (self.output_number_of_channels, self.output_shape, 
                self.output_spacing, self.downsampling_stride, 
                self.kernel_size)

class UpConvConvBlock(nn.Module):
    """
    This is main decoder block. The block is build as follows:
    1.) Upsampling = Transposed Convolution
    2.) Concatenation 

    3.) 3D Conv 
    4.) Insctance normalisation 3D
    5.) Leaky Relu  (negative slope 0.01)

    6.) 3D Conv 
    7.) Insctance normalisation 3D
    8.) Leaky Relu  (negative slope 0.01) (1st output)

    9.) Output_mini_block (2nd output)
        9.1) 1x1x1 Conv
        9.2) Softmax

    """

    def __init__(
        self,
        number_of_input_channels: int = 320,
        number_of_output_channels: int = 256,
        upsample_stride: tuple = (2,2,2),
        kernel_size: tuple = (3,3,3),
        dropout: bool = False,
        number_of_output_classes: int = 7,
        return_auxiliary_output: bool = False):

        """
        Init method of the module. Main building block of Decoder.

        Args:
           * number_of_input_channels, int, number of input channels of the block 
           (for the convolutions number of featres/kernels).
           * number_of_output_channels, int, number of output channels of the convolution operations 
           (output of the block for next block).
           * upsample_stride, tuple(int, int, int), stride for the traspose convolution 
           (upsampling).
           * kernel_size, tuple(int, int, int), kernel size for the convolutions.
           * dropout, bool, if set to True, dropout with p = 0.2 will be applied to
             the both convolutions in the block.
           * number_of_output_classes, int, this is the number of output classes necessary for 
           auxiliary output. It doesn't interfere with the convolutions block. It is necessary only 
           if return_auxiliary_output is set to True.
           * return_auxiliary_output, bool, if set to true the auxiliary output will be returned 
           along with the classical output required for the next convolutional block.
        """

        # Super
        super(UpConvConvBlock, self).__init__()

        # Set return output
        self.return_auxiliary_output = return_auxiliary_output

        # Dropout
        self.apply_dropout = dropout

        # Kernel stride is always 1x1x1 (same)
        _kernel_stride = (1,1,1)

        # Upsample == Transpose convolution
        self.upsample = nn.ConvTranspose3d(in_channels = number_of_input_channels,
                                           out_channels = number_of_output_channels,
                                           kernel_size = upsample_stride,
                                           stride = upsample_stride,
                                           bias = True)
        # Build layers
        self.dropout_1 = nn.Dropout3d(p = 0.2)
        self.dropout_2 = nn.Dropout3d(p = 0.2)

        # First convolution
        self.conv_1 = nn.Conv3d(in_channels = 2 * number_of_output_channels,
                                out_channels = number_of_output_channels,
                                kernel_size = kernel_size,
                                stride = _kernel_stride,
                                padding = [(_i - 1) // 2 for _i in kernel_size],
                                dilation = 1,
                                bias = True)

        self.conv_2 = nn.Conv3d(in_channels = number_of_output_channels,
                                out_channels = number_of_output_channels, 
                                kernel_size = kernel_size, 
                                stride = _kernel_stride,
                                padding = [(_i - 1) // 2 for _i in kernel_size],
                                dilation = 1, 
                                bias = True)

        # Instance normalization
        self.instance_norm_1 = nn.InstanceNorm3d(num_features = number_of_output_channels,
                                                    eps = 1e-05,
                                                    affine = True)
        self.instance_norm_2 = nn.InstanceNorm3d(num_features = number_of_output_channels,
                                                    eps = 1e-05,
                                                    affine = True)

        # Activation function
        self.l_relu_1 = nn.LeakyReLU(negative_slope = 1e-2)
        self.l_relu_2 = nn.LeakyReLU(negative_slope = 1e-2)

        # Output_miniblock
        self.output_miniblock = OutputMiniBlock(
            number_of_input_channels = number_of_output_channels,
            number_of_output_channels = number_of_output_classes)

    def forward(self,
                x : torch.Tensor = None,
                skip_x : torch.Tensor = None)->List[torch.Tensor]:
        """
        Forward function of the upsampling followed concatentation and then 2 conv blocks. 
        The Auxilary output is also calculated here by performing outputMiniBlock
        (Linear to reduce number of channels followed by softmax)

        Args:
            * x, torch.Tensor, output of previous block in the encoder
            * skip_x, torch.Tensor, output of the corresponding skip connection
        
        Returns:
            list but it is exactly 2 torch tensors, first one is output of this layer, second
            one is the auxilary output (if flag is set to true)
        """

        # Upsample
        _x = self.upsample(x)   

        # Concat
        _x = torch.cat((_x, skip_x), dim = 1)

        # Mini block 1
        _x = self.conv_1(_x)
        if self.apply_dropout:
            _x = self.dropout_1(_x)
        _x = self.instance_norm_1(_x)
        _x = self.l_relu_1(_x)
        # Mini block 2
        _x = self.conv_2(_x)
        if self.apply_dropout:
            _x = self.dropout_2(_x)
        _x = self.instance_norm_2(_x)
        _x = self.l_relu_2(_x)

        # Calculate Auxiliary output
        if self.return_auxiliary_output:
            _aux_output = self.output_miniblock(x = _x)
            # Return
            return _x, _aux_output
        else:
            # Return
            return _x, None

class OutputMiniBlock(nn.Module):
    """
    Small block for outputing the number of classes and obtaining final result

    It is:
        1.) linear_layer: 1x1x1 conv to transform n_channel to _number_of_output_classes_channel
        2.) softmax (nothing special)
    """
    def __init__(
        self,
        number_of_input_channels: int = None,
        number_of_output_channels: int = None):
        """
        Init function of the small block that makes output

        Args:
            * number_of_input_channels, int, number of input channels in linear layer
            * number_of_output_channels, int, number of output channels in linear layer
        """
        # Super
        super(OutputMiniBlock, self).__init__()

        # Linear layer
        self.conv_1x1x1 = nn.Conv3d(in_channels = number_of_input_channels, 
                                    out_channels = number_of_output_channels, 
                                    kernel_size = (1, 1, 1),
                                    stride = (1,1,1),
                                    padding = 0,
                                    bias = True)

        # Output softmax
        #self.softmax = nn.Softmax(dim = 1)

    def forward(
        self,
        x : torch.Tensor = None)->torch.Tensor:
        """
        Forward pass,

        Args:
            * x, torch.Tensor, input tensor 
        
        Returns:
            * torch.tensor as an output of targetet classes with applied softmax
        """

        # Get desired number of classes
        _x = self.conv_1x1x1(x)

        # Apply softmax
        #_x = self.softmax(_x)

        # Return
        return _x