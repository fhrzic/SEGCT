# Readme file for the Simplified version of the NN-Unet 


## Augmentations

Augmentations are implemented base on [TorchIO](https://torchio.readthedocs.io/) library and follows the _"batchgenerators"_ augmentations from the NN-Unet implementation. The implemented augmentations are stored in ${\color{green}\text{"Augmentation/ImplementedAugmentations"}}$. 


List of implemented augmentation is as follows:
* <details> <hr> <summary> BrightnessTransform </summary> 
    Class which implements brightness transform on the image. Script which implements it is named: <p style="color:green;">brightness_transform.py</p>
    
    <b>BrightnessTransform applies</b>: image only
    
    <b>Args</b>:
    <ul>
    <li><b>intensity_range</b>, Tupple of two floats, this is a range from which a random number is sampled and then that number
    serves as multiplier of the brightness</li>
    <li><b>synchronize_channels</b>, boolean, if set to True: all channels will have the same multiplier, if False, one multiplier per channel will be sampled from intensity_range</li>
    <li><b>probability_per_channels</b>, float, probability that brightness transform applies to each channel. E.G. for value 0.25 it means that each channel have 0.25 chance to have brightness applied to it.</li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> CenterCrop </summary> 
    Class which applies center crop. This is importat because when we use rotation, if we perform it on the exact shape that the network is expected to get as the input, part of that shape will get outside the selected patch boundaries due to the rotation. So to prevent content loss and unecessary padding with zeros, we sample bigger patch, perform all necessary operations on it, and then we extract only the center part which contains the region of interest (with most usefull pixels) as possible. Script which implements it is named: <p style="color:green;">center_crop.py</p>
    
    <b>CenterCrop applies</b>: both image and label
    
    <b>Args</b>:
    <ul>
    <li><b>desired_shape</b>, list of integers, arround centre voxel of the provided patch, this shape will be cropped out by simetrically dividing the desired shape to two parts. E.G. 100, 100, 100 will crop 50 from the bot, top, left, right, front, back from the centre voxel</li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> ContrastTransform </summary> 
    Class which implements random brigthness transform per channel of the image. Script which implements it is named: <p style="color:green;">contrast_transform.py</p>

    <b>ContrastTransform applies</b>: image only
    
    <b>Args</b>:
    <ul>
    <li><b>contrast_range</b>, Tupple of two floats, this is a range from which a random number is sampled and then that number
    serves as multiplier of the contrast</li>
    <li><b>preserve_range</b>, boolean, if set to true, the minimum and maximum of the image is preserved after the contrast is
    applied. This is done by clamping the values to the values before applying the contrast.</li>
    <li><b>synchronize_channels</b>, boolean, if set to True: all channels will have the same multiplier, if False, one multiplier per channel will be sampled from contrast_range</li>
    <li><b>probability_per_channels</b>, float, probability that contrast transform applies to each channel. E.G. for value 0.25 it means that each channel have 0.25 chance to have contrast transform applied to it.</li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> DeepSupervisionResize </summary>  
    If we are using deep supervision, this means that we need to provide many different scales of the output segmentation map. Well, this is exactly what this augmentation do so it is always called last. For the provided shape list it simply resize it by using nearest-exact value so we always stay within the same value space. Script which implements it is named: <p style="color:green;">deep_suprevision_resize.py</p>
    
    <b>DeepSupervisionResize applies</b>: label only

    <b>Args</b>:
    <ul>
    <li><b>shapes_list</b>, List of list where every sub-list is having 3 integers representing dimensions, this augmentation resizes the output segmenetation mask to the shapes given in list. This shapes can be obtained from the model. E.G. [[128,128,128], [64,64,64]] will return list with two segmentation maps with given shape. The list is not actually returned, it populates subject property named "ds_labels" (TorchIO agumentation framework for more info). The rescaling is done as said with nearest-exact value in torch.nn.functional interpolate function</li>

    <li><b>p</b>, float, probability to apply this transform. It should be 1 or 0 for this transform. </li>
    </ul>
  <hr> 
  </details>
* <details> <hr> <summary> GammaTransform </summary> 
    Class which implements random gamma transform per channel of the image. Gamma transform is actually gamma correction - process of adjusting the brightness of an image. Script which implements it is named:<p style="color:green;">gamma_transform.py</p> 
    
    <b>GammaTransform applies</b>: image only

    <b>Args</b>:
    <ul>
    <li><b>invert_image</b>, bool, if set to true, the colour of the image will be inverted (multiplyer by -1 because the image is normalized around 0). </li>
    <li><b>synchronize_channels</b>, boolean, aprobability that gamma transform applies to each channel. E.G. for value 0.25 it means that each channel have 0.25 chance to have gamma transform applied to its. </li>
    <li><b>retain_stats</b>, bool, if True, the mean and std will be preserved when transfered to new "gamma multiplied" space </li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>   
  <hr>
  </details>
* <details> <hr> <summary> GaussianBlur </summary> 
    Class which implements gaussian blur in the NN-Unet style. It is possible to have   gaussian blur from TorchIO, but it is not flexible as much as this impleentation. So how it works: p is a general probability that 
    this transfrom will occure. Let's say it is 0.5, it means there is 0.5 chance that 
    this probability will happen. Now, the gaussian blur is applied per each channel
    independently with chance probability_per_channel. This means that although this 
    augmentation is trigered and is applied to the input tensor, it is possible that 
    no transform occures because of the probability_per_axis is not satisfied (chance returned true). Regarding sigmas, sigma dictatate how much blur is being applied. It can be same across all the channels (synchronize_channels), and then within the channel it can be same accross all the axes(synchronize_axes). Also, the oposite applies (different for each channel and different for each axis). Script which implements it is named <p style="color:green;">gaussian_blur.py</p>

    <b>GaussianBlur applies</b>: image only

    
    <b>Args</b>:
    <ul>
    <li><b>sigma_range</b>, tuple of two floats, the sigma will be sampled from this range for each axis. Bigger sigma more spread out kernel, smaller sigma will result in narrower kernel, This basically means how severe blur will be. </li>
    <li><b>probability_per_channel</b>, float, probability that low resolution applies to each channel. E.G. for value 0.25 it means that each channel have 0.25 chance to have low resolution applied to its.  </li>
    <li><b>synchronize_channels</b></li>, boolean, if true, all channels will have same sigma, if false, for each channel different sigma will be sampled
    <li><b>synchronize_axes</b>, boolean, if true, all axes inside the same channel will have the same scale. I know, this can be confusing. But if we have 3D 3 channel image, we can have 6 possible sigmas. If we set synchronize_channels to True, we still can have 3 different sigmas for 3 axes if we chose synchronize_axes to be False. But every channel will have those same 3 sigmas for its axes. This is almost never the case (CT is 1 channel 3D volume) </li>
    <li><b>p</b>, float, probability to apply this transform. So if we have 1 channel, and probability_per_channel is 0.5, with p = 1 we still have only 0.5 chance to apply the augmentatation (1 * 0.5) </li>
    </ul> 
    <hr>  
  </details>
* <details> <hr> <summary> LowResolution </summary>  
    Class which implements random low resoltuion. How it does it? 
    Select_channel_of_iamge--> choose random scale per axis -->interpolate to that randomly chosen. 
    dimension --> interpolate back to original dimension. Script which implements it is named
    <p style="color:green;">low_resolution.py</p>

    <b>LowResolution applies</b>: image only

    <b>Args</b>:
    <ul>
    <li><b>scale_range</b>, tuple of two floats, this is the range from which the low resolution scale will be sampled</li>
    <li><b>synchronize_channels</b>, boolean, if set to True: all channels will have the same scale, if False, one scale per channel will be sampled from scale_range </li>
    <li><b>synchronize_axes</b></li>, boolean, if true, all axes inside the same channel will have the same scale. I know, this can
    be confusing. But if we have 3D 3 channel image, we can have 6 possible scales. If we set synchronize_channels to True, we still can have 3 different scales for 3 axes if we chose synchronize_axes to be False. But every channel will have those same 3 scales for its axes. This is almost never the case (CT is 1 channel 3D volume)
    <li><b>probability_per_channel</b>, float, probability that low resolution applies to each channel. E.G. for value 0.25 it means that each channel have 0.25 chance to have low resolution applied to its. 

    <li><b>p</b>, float, probability to apply this transform. So if we have 1 channel, and probability_per_channel is 0.5, with p = 1
    we still have only 0.5 chance to apply the augmentatation (1 * 0.5) </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> RemoveLabel </summary>  
    Class which implements remove label. Well it does not actually remove it, it  is more like it "replace it" with another label. This is usefull when dealing with ignore labels and want to converte them to background. Script which implements it is named
    <p style="color:green;">remove_label_transform.py</p>

    <b>RemoveLabel applies</b>: label only

    <b>Args</b>:
    <ul>
    <li><b>source_label</b>, int, valid label that we want to change </li>
    <li><b>destination_label</b>, int, valid label to which we want to change the source_label to </li>
    <li><b>p</b>, float, probability to apply this transform. Honestly, it is either 0 or 1, other does not
    make the sense - but I am not forcing it. </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> Rotation </summary> 
    Class which implements random rotation on both image and segmentation map. Script which implements it is named
    <p style="color:green;">rotation.py</p>

    <b>Rotation applies</b>: label (nearest interpolation) and image (bilinear interpolation)

    <b>Args</b>:
    <ul>
    <li><b>rotation_range</b>, list of two integers, the random angle will be chosen from the range defined by the 
            two provided integers. The angles are given in <b>degrees</b>; E.G [-30, 30] will chose a random angle between
            -30 and 30. Same angle applies to both label and image </li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>
  <hr>
  </details>
* <details> <hr> <summary> Scaling </summary>
    Class which implements random scaling on both image and segmentation map. Script which implements it is named
    <p style="color:green;">scalling.py</p>

    <b>Scalling applies</b>: label (nearest interpolation) and image (bilinear interpolation)

    <b>Args</b>:
    <ul>
    <li><b>scale_range</b>, Tuple of two floats defining the minimum and maximum scale factors. E.G (0.7, 1.5) means that the
    random scale will be choosen from a given interval. </li>
    <li><b>apply_same_ratio</b>, boolean, if set to True, same scale will be use along all axes. E.G. (True), same random scale
    sampled from the scale_range will be applied to every axis. If (False), for each axis one scale will be sampled from scale ratio</li>
    <li><b>p</b>, float, probability to apply this transform. </li>
    </ul>
  <hr>
  </details>

List of additionally used augmentations provided by TorchIO:
* <details> <hr> <summary> RandomNoise </summary> 
  TorchiIO implementation: <a href="https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomNoise">link_to_doc</a>.
  <hr>
  </details>
* <details> <hr> <summary> RandomFlip </summary>  
  TorchIO implementation: <a href="https://torchio.readthedocs.io/transforms/augmentation.html#torchio.transforms.RandomFlip">link_to_doc</a>.
  <hr>
  </details>

Although some of the impemented augmentation methods exists in TorchIO framework and are already implemented, their behaviour
and flexibility was not appripriate or sufficient for usage in this project. Hence, there were replicated to mirror the augmentations
used in the NNUnet (batchgenerators).

---
#### TorchIO augmentation framework

Here I want to present logic behind TorchIO framework - at least the part how I am using it and for what I have used it in code.
If someone want to implement new augmentation in TorchIO, it needs to understand following components:

1. <details> <hr> <summary> Input data </summary> 
   TorchIO works with a dictionary type called <b>tio.Subject</b>. It typically have two keys: <i>image</i> and <i>label</i>, which in our case is ct-scan (patch of a a volume) for the image, and the corresponding segmentation map for the selected patch as a label. I decided to add one more key which is called <i>ds_labels</i>. In case that we use <b>DeepSupervisionResize</b>, the resized segmentation maps will be stored in <i>ds_labels</i>. For the values, they are <i>torch.tensors</i> for both image and label. A code snippet of a input data is as follows:
   <pre>
    <code>
    subject = tio.Subject(
        image = tio.ScalarImage(tensor = input_patch),
        label = tio.LabelMap(tensor = segmentation_mask),
        ds_labels = []
    )
    </code>
   </pre>

   In the code snippet it can be noticed the <i>tio.ScalarImage</i> and <i>tio.LabelMap</i> "contatiners". Well, they are important in the next step.
   <hr>
   </details>
2. <details> <hr> <summary> How it works </summary> 
    Each class which implements transformation (augmentation of any type) inherits <i>tio.Transform</i> class. This parent class makes our implementation compatible with TorchIO ecosystem. Every transformation has mandatory <b>init method</b> and <b>apply_transform</b>:
    
    * <b>init method</b> accepts arguments explained in the chapter above. Basically it defines the transformation.
    * <b>apply_transform</b> method which implements the logic of the transformation and applies it. It accepts and returns  <i>tio Subject</i> as mentioned into "Input data" section.

    It is also usefull to understand the folowing implementation details:
    * <b>tio.ScalarImage</b> and <b> tio.LabelMap </b> - tio containers for tensors. ScalarImage represents scalars while LabelMap represents categorical labels (segmentation maps). In the code,  image and label are set to be of type torch.tensor which is generated from the loaded seg map or ct. For more info visit <a href="https://torchio.readthedocs.io/data/image.html">link_to_doc</a>. 
    
    * <b>tio.Compose</b> - if we want to make a chain of augmentation to apply, we can provide a list of augmentation and give it to <i>tio.Compose</i>. This makes them connected in a pipeline so we have only one "method" to call. Every transformation in pipline has it owns probabiliy <b>p</b> to be applied. 
    * <b>tio.Lambda</b> - it accepts tio.Compose as argument and has argument  <i>types_to_apply</i> which forces augmentations only to be applied to, for instance, tio.IMAGE (equivalent to tio.ScalarImage). Basicaly, it is and extra safety step which forces pipline to be applied only to images. It can also be applied to the labels. Hard to debug... 
   <hr>
   </details>
<hr>

### How to use augmentation

To use priviously described augmentations in the way NNUnet uses them, I have created class which wraps the selected augmentations in a single pipeline. The concept is very simple: we have config file in ${\color{green}\text{"Configurations/config\_augmentation.py"}}$ which contain a class named <b>AugmentationConfigNNUnet</b>. This class have all parameters for each augmentation + one extra parameters which indcate if selected augmentation applies. E.G for Rotation we have:
```
# Random affine, rotation, scaling, and mirror
a_rotation = True # Indicator if rotation will be applied
a_r_degrees = (-30, 30) # (a,b), uniformly chosen angle from a to b
a_r_probability = 0.2 # Probability to apply
```
<i>a_rotation</i> indicates that the rotation transform applies. This configuration is implemented as a class and has three methods:

1. <b>\_\_str\_\_</b> - simply prints out all augmentation active (statuses of extra paramters which indicates if augmentation applies)
2. <b>export_to_json</b> - method which accepts <i>export_path</i> as argument where the whole augmentation config will be stored in a ".json" format.
3. <b>import_from_json</b> - method which accepts <i>import_path</i> as argument and loads the stored augmentation paramters to the class. It is not used anywhere in the code, but it seems as a nice feature to have.

<hr>

Now when we defined all the paramteres and desired augmentations, we have to actually build the augmentation method. In
${\color{green}\text{"Augmentation/Utils/nn\_unet\_augmentation.py"}}$ is a class <b>NNUnetAugmentation</b> which accepts as argument the config class defined before. Based on the provided config class it builds the pipeline and connectes all required augmentations together. This class contain four methods:

1. <b>\_\_init\_\_</b> - method which builds augmentation piple based on the provided config file. 
2. <b>\_\_call\_\_</b> - this is a call method which accepts <i>input_patch</i> and <i>segmentation_mask</i> on which it perform desired augmentations.
3. <b>__calculate_initial_patch_size</b> - private method which calculates the required input size of a patch so the desired patch can be cropped out of the volume with maximum conetent preserved. This part is explained in the rotation. One more time, if we rotate/scale the desired shape, we will need to zero pad a lot, so to prevent extencive zeropadding, we sample bigger patch and then crop the desired patch out of it after rotation.
4. <b>get_initial_patch_size</b> - returns caluclated inital patch size which is a necessity for datasets scripts where the patch is being sampled.

<hr>

### Implement new augmentation

To implement new augmentation, one can follow the script ${\color{green}\text{"Augmentation/Utils/nn\_unet\_augmentation.py"}}$ and alter the config file in ${\color{green}\text{"Configurations/config\_augmentation.py"}}$. Then when building dataset configuration it must be provided with its own augmentation method. As long all the mentioned methods are present in the scripts, it should work.

<hr>




