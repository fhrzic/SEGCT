# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Models"}}$

As it can be seen, the <span style="color:green;"> "Models"</span> contain sub directory <span style="color:green;"> "SNNUnet"</span> which contains following scripts

```
    Models:
        --> SNNUnet:
            --> building_blocks.py
            --> generate_SNNUnet_fingerprint.py
            --> SNNUnet.py
        --> VNet
            --> V_net.py
            --> config_model.py
        --> inspect_model.py
        --> autograd_and_gradient_clipping.py

```
<span style="color:red;"> VNET model is not tested, it is just implemented that is why it has it own config_model file.</span>.
Config file for the SNNUnet is on standard location for all configs which is <span style="color:green;"> "Configurations/config_model.py"</span>. Regarding the scripts in the SNNUnet, they will be elaborated in following sections.


---

## SNNUnet
Logic behind the scripts in the <span style="color:green;"> "SNNUnet"</span> directory is as follows:

*  <b>generate_SNNUnet_fingerprint.py</b> script which generates fingerprint which is necessary for building the model config file. Fingerprint contains estimation of the model input patch size.

* <b>building_blocks.py</b> contains all the building blocks necessaray to build the NN-Unet.

* <b>SNNUnet.py</b> implementation of the moodel. It combines all the building blocks into one "whole".


However, to undestand how to build NN-Unet, or in this case SNNUnet, it is needed to dig into the config file avalable at 
<span style="color:green;"> "Configurations/config_model.py"</span>.

---

### config_model.py script
<details>

<span style="color:green;"> "config_model.py"</span> located in <span style="color:green;"> "Configurations"</span> directory contains all the necessary attributes/properties to build NN-Unet model architecture. If one would like to use any other model architecture, it would not be this complicated. The attributes in the model are as follows:


<b>Attributes:</b>
<ul>
    <li><b>"model_name"</b> -- string -- later on in training methods, based on this name it is determined which model to invoke. For SNNUnet it is <b>SNNUnet</b> (which is also the default value).</li>
    <li><b>"number_of_input_channels"</b> -- integer -- number of input channels of the data. Defualt is set to 1.</li>
    <li><b>"number_of_output_classes"</b> -- integer -- number of output classes. Default is set to 16.</li>
    <li><b>"input_shape"</b> -- tupple of 3 integers -- this is input shape of the SNNUnet. If path to the data fingerprint is provided, then the input shape can be calculated based on the spacing and provided median shape of the data. Default value is 128x128x128.</li>
    <li><b>"input_spacing"</b> -- tupple of 3 integers -- input spacing of the data. It is important for the calculation of input shape of the neural network and for building the whole topology. If path to the data fingerprint is provided it is extracted then from it. Default value 1x1x1.</li>
    <li><b>"z_last_channel"</b> -- bool -- if true, the z-axis is the last so we have x,y,z. If set to false, we have z,x,y scenario. It is really important to set this paramter right due to the model generation. If the path to the data fingeprint is provided, this parameter will be extracted from it. Default is set to True</li>
    <li><b>"max_batch_size"</b> -- integer -- determined max batch_size which can be processed on given gpu respecting the set gpu memory limitations. Default 2.</li>
    <li><b>"output_shapes"</b> -- list of lists -- output shapes of each "upsample" output in the UNet architecture. It is list of list where every "inner" list are three integers representing shape of the output volume. This is useull to have for augmentation methods.</li>   
    <li><b>"output_spacings"</b> -- list of lists -- same as output_shapes, but it have estimated spacing for every "upsample" output. I think this is not used anywhere, but for the consistency reasons it is good to have it.</li>
    <li><b>"output_auxiliary_predictions"</b> -- bool -- </li>
    <li><b>"device_limit"</b> -- integer -- limit of the device in <b>megabytes</b>. Sometimes we don't want to use the whole device or the maximum patch size. If we want to avoid that - we control it by setting the device limits.</li>
    <li><b>"mixed_precision"</b> -- bool -- if set to true, the model weights will be casted to 16-bits instead of 32 bits which <b>halves</b> the model size enabeling the bigger input patch sizes.</li>
</ul>

<br>

Many of the required attributes can be calculated or extrapolated from the dataset fingerprint file. That is why this implementation of 
<span style="color:green;">ModelConfig</span> class is also having init method. Here is the overview of the methods implemented in the class.

<b>Methods:</b>
<ul>
    <li><b>"__init__"</b> -- method which calculates all the attributes above if <b>path_to_data_fingerprint</b> and <b>device</b> are provided. It also accepts <b>device_limit</b> and <b>mixed_precision</b> as inputs which infulence the size of the input patch a lot! How is it done? By utilising <span style="color:green;">GenerateSNNUnetFingerprint</span> class implemented in <span style="color:green;">Models/SNNUnet/generate_SNNUnet_fingerprint.py</span></li>
    <li><b>"__str__"</b> -- method whcih prints all the parameters stored in the model confing.</li>
    <li><b>"__iter__"</b> -- this is a neat way to give the instance of the class as the argument to the method which builds the model where everything will be filled up automatically.</li>
    <li><b>"export_to_json"</b> -- it exports all properties to the json file defined by the given argument <b>expor_path</b>.</li>
    <li><b>"import_from_json"</b> -- it imports all properties from the json file defined by the given argument <b>import_path</b>.</li>

</ul>

</details>

---

### generate_SNNUnet_fingerprint.py script

<details>
The script <span style="color:green;">generate_SNNUnet_fingerprint.py</span> is located at <span style="color:green;">Models/SNNUnet/</span> directory and it implements the class <b>GenerateSNNUnetFingerprint</b> which creates and calculates the following attributes needed to model generation:

* Input Patch Shape
* Max Batch Size
* Output shapes
* Output spacing
* Z_last_channel (as a bool)
* Number of channels (as an int)
* Number_of_output_classes
* Input spacing

The meaning of each attribute is explained <b>config_model.py</b> and in the section <b>config_model.py script</b> of this readme. Here is the list of methods which are implemented and how each component participate in the final goal of creating the fingerprint which is the cornerstone of the model building.

<b>Methods</b>

<ul>
    <li><b>"__init__"</b> -- Method which prepare and caluclate all necessary properties. It accepts four
    arguments:
    <ul>
    <li><b>"dataset_fingerprint_path"</b> -- string -- path to the datafingerprint generated by the <b>generate_dataset_fingerprint.py</b> script.Namely, that fingerprint file contains all necessary information to build up NNUnet model.</li>
    <li><b>"device"</b> -- string --  name of the device where the model will be trained. Typically it is cuda:0, cuda:1 etc. This is important for estimation of the maximum batch size, but it can be constraind with device_limit.</li>
    <li><b>"device_limit"</b> -- integer -- limit set in <b>MB</b>to which the device will be used. This is calculated for batch size of 2. which means that when the method is estimating the model size, it is always estimated for batch size of 2. And then it can be increased to the maximum device size if necessary. Logic is the following: We want to fit 1 foreground and 1 background sample on the GPU, so batch size is 2. Device limit is the amount of resources we are willing to spent on this small batch. And then we know that if have 48GB, we set limit to 12GB, we can have the batch size of 8 (4*2) while the patch size is going to be maximum one that can be fitted.</li>
    <li><b>"mixed_precision"</b> -- bool -- If set to true, the model weights and computation will be performed in 16bit precision and not 32bits which saves a lot of space on the GPU.</li>
    </ul>
    This method first obtain all necessary informations from provided fingerprint. Then it checks if z-channel order is regular(integer because it is calculated as a mean of all z-channels in dataset), and if it is it caluclates if it is first or last. Also it checks if number of channels is integer (if it is not, please check the dataset and fingerprint file... because this means not all images have the same number of channels). Then it calls methods <b>__obtain_candidate_shapes</b> and <b>__find_first_combination_that_fits</b> to estimate the input shape. The first method estimates the shapes and create a list of possible shapes based on set policies, while second one seeks for first/largest combination which fits on the selected <b>device</b>.
    </li>
    <li><b>"__obtain_candidate_shapes"</b> -- First it estimates the biggest possible shape <b>median_spacing</b> by utilizing formula  
    presented in code (inherited from NNUnet code). The intial shape is than clamped to the maximum intial <b>median_shape</b>. Then it calculates number of "halvings" by calling method <b>__perform_havling_nnunet_way</b>. Havling calculates number of cascades in the UNet topolgies; to precise number of downsamplings per each dimension of the image. Exact policy is given in explanation of __perform_havling_nnunet_way method. After havliung we find next divisible by calling <b>__next_divisible</b> method which gives us the new candidate patch shape. Here we do not reduce all dimensions at the same time, rather we reduce only the biggest dimension in relatio to the inital shape. That reduction, the bigest one, is reduced by the 2**<i>of number of halvings</i> for that shape. This is performed until <b>estimated patch shape is greater than (8,8,8)</b>. This is only to get the list of the potential candidates - rough estimation. The real candidate is selected by <b>__find_first_combination_that_fits</b> method.
    </li>
    <li><b>"__perform_havling_nnunet_way"</b> -- method which simulates encoder of the NNUnet. It does not performe convolution but it halves each dimension by halving it under ceil operation. Then if dimension is higher than 4, the halving for that dimension is permited. Otherwise it is not. It returns number of halvings per each dimension which is a requirement to calculate the reduction factor for next valid shape.</li>
    <li><b>"__next_divisible"</b> -- method which calculates the next divisible of x by the 2**y.  It is used to estimate new patch size based on the calculated number of halvings.</li>
    <li><b>"__find_first_combination_that_fits"</b> -- method which performs binary search over the sorted candidate list for the first candidate which fits in the device respecting the set limit. to do so, it uses method <b>__estimate_model_size</b> which simulates SNNUnet based on the input spacing, given patch size, number of classes... it simply pretends that it is now building a real neural network and performs forward/backward pass calculating how much MB it costs. Again, after performing the search, index of the patch size that did fit in the selected device is retured as the biggest patch size that can be given to the NNUnet model (which is generated based on that patch).</li>
    <li><b>"__estimate_model_size"</b> -- method which esxtimates size of the model by performing forward and backward pass. Auxiliary 
    predictions are set to true because they generate negligable amount of MB to the model (but they still do). Estimation is done by utilizing <b>torchinfo.summary</b> (<a href="https://github.com/TylerYep/torchinfo">torchinfo</a>). Unofrutnatelly... the generated summary must be parsed because it does not offer export to dir to extract the model size. This parsing is done in <b>__extract_estimated_size</b> method.</li>
    <li><b>"obtain_fingerprint"</b> -- the only method which is intend to be called in this class by the user. As the name says, it obtains the fingerprint to build model. It return the dict, but it can be as well exported to the <b>export_path</b> which is provided as the argument to the method. It obtains </li>
    <li><b>"__estimate_max_batch_size"</b> -- method which calculates the max batch size which can be put on the given device. There are two constraints for max batch size:
    <ul>
    <li>must fit in the device</li>
    <li>must not exceed 5% all pixels in the training set. (This is for small datasets)</li>
    </ul></li>
    <li><b>"__obtain_output_shapes_and_spacings"</b> -- method which obtains output shapes for auxilary losses and spacings at each stage of the model.</li>
    <li><b>"__extract_estimated_size"</b> -- method which uses regex to extract model size from provided summary.</li>

</ul>

At the end, what the script produces is fingerprint for the model where the most important information is <b>input patch size</b> that can be placed on the device with <b>batch size 2</b>.
</details>

---

### building_blocks.py script
<details>

Building blocks as the name says are the building blocks used to build NN-Unet. There are three building blocks that helps to build the network:

<ul>
<li>ConvConvDownBlock</li> -- as the name says, it is a nnModule which is the main building block of the <b>encoder</b>. It consists of: 
<ul>
<li>Convolution layer</li> -- this is a <b>"smart" 3D convolution</b> with adaptrive stride and kernel shape. Kernel can be 3x3x3 where any 3 can be swaped by 1(e.g. 3x1x1) based on the spacing. Namely, if spacing is 2 times bigger than the minimum spacing in that layer, then that dimension which is 2x<i>minimum_spacing</i>. Stride is always set to 1x1x1. The idea is to adapt kernel filters based on the spacing of the input volume that that convolution sees. Number of kernel filters increases from layer to layer by factor of 2 until maximum of 320. 
<li>Dropout layer</li> -- dropout layer with p=0.2.
<li>Instance normalization</li> -- instance normalization with eps = 1e-05 and affine set to true(learnable parameters).
<li>Leaky relu</li> -- leaky relu activation function with negative slope set to 1e-2.
<li>Convolution layer</li> -- same as previous one with exception of number o filters. Here the number of input filters is equal to the number of output filters, while in previous convolution the numer of input filters was set to be the number of output filters of the previous step.
<li>Dropout layer</li> -- same as previous one.
<li>Instance normalization</li> -- same as previous one.
<li>Leaky relu</li> -- same as previous one.
<li>strided convolution (downsample)</li> This convolution is same as the previous convolutions, but it now have stride different so it downsample only required dimensions. Requried dimension is any dimension greater than 4 or its spacing is less than double from the minimum stride.
</ul>

<li>UpConvConvBlock</li> -- this layer is the reverse from the ConvConvDownBlock which builds the neural network <b>decoder</b> which is consisted of the following blocks:
<ul>
<li>strided convolution (upsample)</li> This convolution is same as the previous convolutions, but it now have stride different so it upsample only required dimensions. Required dimension is caluclated based on the corresponding encoder layer so that the concatenation of the next layer can be performed. 
<li>Concatenation block</li> -- concatenation block concatinates the output of upsampling in the decoder with the output of the corresponding layer in the encoder.
<li>Convolution layer</li> -- Classical 3D convolution with kernel stride set to 1x1x1 and kernel size set to the correspoinding number of kernels calculated for the cooresponding encoder block. 
<li>Dropout layer</li> -- dropout layer with p=0.2.
<li>Instance normalization</li> -- instance normalization with eps = 1e-05 and affine set to true(learnable parameters).
<li>Leaky relu</li> -- leaky relu activation function with negative slope set to 1e-2.
<li>Convolution layer</li> -- same as previous one with exception of number o filters. Here the number of input filters is equal to the number of output filters, while in previous convolution the numer of input filters was set to be the number of output filters of the previous step.
<li>Dropout layer</li> -- same as previous one.
<li>Instance normalization</li> -- same as previous one.
<li>Leaky relu</li> -- same as previous one.
<li>Aux output</li> -- this block also provides auxiliary output requred for deep supervision loss. The output of the block, if it is requiered is passed to <b>OutputMiniBlock</b>
</ul>
<li>OutputMiniBlock</li> -- is a special blocks which takes the input volume, applies 1x1x1 convolution on it in order to reduce the dimension of it to the number of classes and that way makes it adaptable for prediction. It does not apply any nonlinear functions like Softmax on the output. This must be applied in the loss functions if wanted. It produces pure logits.
</ul>
</details>

---

### SNNUNet.py script

<details>
Main script for building model. It is comprised of three classes:

<ul>
<li>SNNUnet</li>
-- Main class for building model. This model puts together <b>encoder</b> and <b>decoder</b> part of the network and it accepts following arguments:
<ul>
<li>input_shape</li> -- tuple of three integers -- represents the input shape of to the model.
<li>input_spacing</li> -- tuple of three FLOATS -- represents the input spacing of the model.
<li>number_of_channels</li> -- integer -- represents number of input channels
<li>z_last_channel</li> -- boolean -- if z channel is the last or the first one (set to True if it is the first one).
<li>number_of_output_classes</li> -- integer -- number of classes to predict. This is important for output layers.
<li>output_auxiliary_predictions</li> -- boolean -- if set to True, every upsampling will also produce the prediction on that level of resolution.
</ul>

How it works? It invokes encoder which creates encoding branch of the neurla network. The encoder has a cool method called <b>obtain_fingerprint</b> which returns number of filder, shapes and spaces for each level of the encoder (by level i mean cascade). Whith this information, we build the decoder. As any nn.Module it has <b>forward pass method</b> which implements the described logic. It also have method <b>get_output_shapes</b> which  returns dict containing shapes and spaces. Shapes and spaces are sorted 
from the first layer of the decoder to the last one (from last output to the bottlneck, full resolution to the lowest resolution). The encoder returns them in the oposite direction. This is usefull for deep supervision loss.

<li>Encoder</li> -- encoder branch of the model. Logic is that it utilize <b>ConvConvDownBlock</b> which is basicaly one cascade. The beauty of this class is that it also calculates everything necessary to build this branch including number of cascades. The arguments are as follows:

<ul>
<li>input_shape -- tuple of three integers -- input shape to the encoder.</li>
<li>input_spacing -- Tuple[float, float, float] -- input spacing to the encoder.</li>
<li>number_of_channels -- integer -- number of input channels. </li>
<li>initial_features_number -- integer -- number of the features (kernels) in the first cascade. Every subsequent cascade will have this number doubled until the <i>feature_num_cap</i> is reached. Then <i>the feature_num_cap</i> is used.</li>
<li>dropout -- bool -- use dropout or not.</li>
<li>feature_number_cap -- integer -- maximum number of features in a cascade </li>
<li>return_skip_layers -- bool -- this is always true, but maybe someone want to just use encoder and don't care about skip connections so there, this is also an option. </li>
<li>z_last_channel -- bool -- is the z last channels (important for stopping the number of cascades - additional constraint on the cascade building)</li>
</ul>

How does it works? Easy, each "cascade" is actually a <b>miniblock</b> consisted of <b>ConvConvDown</b> which is explained in the paragraph above. This encoder calculates the inputs for each of the miniblock by taking care of shapes, spacings, number of filters. It also have constraints for how long it can do the downsampling of the blocks (until all shapes are not less than 5, or until in-plane shapes are not equal to 8 -- z-channel important to determin in-plane indexes. REMEMBER downsampling is dependant on spacing). All block are calcuallated in the <b>\_\_init\_\_ method</b>. Forward pass is implemented in <b>forward method</b> where the input volume is passed through all caluclated blocks. There is another method named <b>obtain_fingerpring</b> which returns all strides, convolution kernels, features numbers, shapes and spacings for each of the cascade which make building <b>decoder</b> part of the model feasiable.

<li>Decoder</li>
-- Decoder class builds decoder branch and it accepts most of the the parameters obtained by <b>obtain_fingerpring</b> method of the Encoder class along with skip connections (outpus) of the encoder at each cascade. Here are the arguments:

<ul>
<li>upsampling_stride_list -- list of tupples where each tupple has 3 integers -- to build up decoder, it is necessary to know the strides how the downsampling was done because the shapes must match for the concatenation layer. This list is obtained from encoder.</li>
conv_kernels_list-- list of tupples where each tupple has 3 integers --  since not all conv kernels are 3x3x3 due to the spacing, it is necessary to perfrom the same convolution in the decoder as it was in the encoder. And that is exactelly what is done here. The exact list of kernel sizes is obtrain from encoder.</li>
features_list -- list of integers -- number of features (kernels number) for each convolution obtained from encoder.</li>
output_auxiliary_predictions -- bool -- if set to true, the auxiliary predictions are caluclated and outputed via <b>OutputMiniBlock</b> at the end of each cascade in the decoder</li>
number_of_output_classes -- integer -- number of classes to which the final output is being reduced.</li>
</ul>

How dose it works? Well, encoder "shrinks" while decoder "expands" the features maps serving as the Encoders counterpart. To do it properly, all necessarry information is obtained by the method <b>obtain_fingerpring</b> in the <b>Encoder</b> class. Based on the obtained information the <b>\_\_init\_\_</b> method from <b>Decoder</b> which utilizes <b>UpConvConvBlock</b> (in consequtive manner) to create sequnetial branch of the model. Now in <b>forward</b> method is the main difference between Encoder and Decoder. Namely, forward method accepts argument <b>skip_connections</b> whcih are outputs of the encoder branch before each downsampling. This are the feature vectors which are conacatinated to output of each <b>UpConvConvBlock</b> after upsampling. Lastly, after each cascade the auxiliary output is caluclated with the help of <b>OutputMiniBlock</b>. At the end, encoder reutrns either <b>single prediction map</b> or <b>list of prediction maps</b> for each cascade.
</ul>

</details>

### autograd_and_gradient_clipping.py script

<details>
In this script two classes are implemented which deals with GradScaler from torch. GradScaler isIf we want to use it on cuda, well I just followed the official instructions:

<ul>
<li>GradScaler</li> -- class which inherits and implements GradScaler from torch. Please visit <a href="https://pytorch.org/docs/stable/_modules/torch/cpu/amp/grad_scaler.html#GradScaler">troch_modules</a> for more info.
<li>DummyContext</li> -- to be convinient, a class which just serves as a placeholder if someone decided to not use gradient scaling.
<ul>

</details>

---

### inspect_model.py script

<details>

Script which helps to inspect the models. It is not called anyway but it showes how to create model confing, save it, load it, and print the summary of the model for the further inspection. The script takes three arguments, but it is here to help see the model:

<b>ARGS</b>:
<ul>
<li>data_fingerprint_path -- string -- path to the dataset fingerprint generated by <span style="color:green;">DatasetFingerprint/generate_dataset_fingerprint.py</span> script. Based on information presented ".json" file the model is built.</li>
<li>device -- string -- name of the device where the model will be set E.G. "cpu", "cuda:0", "cuda:1"...</li>
<li>export_dir -- string -- directory to which <b>model_config.json</b> and <b>model_inspection.txt</b> will be saved. Json file holds the confing to build the model, while ".txt" file hold build model in a way that it can be read by human.</li>
</ul>
</details>