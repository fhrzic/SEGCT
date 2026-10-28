# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Losses"}}$

As it can be seen, the <span style="color:green;"> "Losses"</span> contains following scripts:

```
    Losses:
        --> compound_loss_DC_and_CE.py
        --> deep_supervision_loss.py
        --> soft_dice_score.py
```

Config file for the Lossies is on standard location for all configs which is <span style="color:green;"> "Configurations/config_loss.py"</span>. 


---

### config_loss.py script
<details>

<span style="color:green;"> "config_loss.py"</span> located in <span style="color:green;"> "Configurations"</span> directory contains all the necessary attributes/properties to build implemented losses. In total three losses are implemented from whose two are in use during the SNNUnet model training. The implemented losses are as follows:

<ul>
<li>LossConfigFromTorch -- confing which serves as config for any loss already implemented in torch. The confing file is the one that is provided to training script, but it must contain all necessary information to build losses. In this case, if loss is implemented in Torch, it is only needed to provide its name (<b>loss_name</b>), and then in training script invoke the method implementing that loss. If we need more parameters, they can be provided. This is the simplest class for any loss.</li>
<li>LossConfig_DCandCE -- config for the Dice and Cross entropy loss. It contains many parameters necessary to build this complex loss which is created from two well know losses. Here is the parameter brakedown:
<ul>
<li>loss_name -- name of the loss so the training script knows which loss to cast. It must be a unique name.</li>
<li>apply_nonlin -- applies to SoftDiceScore. It is a callable function which is applied to the exit of the neural network. Remember, output of the SNNUnet are logits.</li>
<li>batch_dice -- applies to SoftDiceScore. If true, the loss will be caluclated by summing all TP, TN, FN, across batches. If False, for each sample in batch loss will be calculated, and then mean of the calculated losses will be taken as a ground truth.</li>
<li>do_bg -- applies to SoftDiceScore. If True, background (0) is also treated as a class.</li>
<li>smooth -- applies to SoftDiceScore. Necessary to avoid division by 0.</li>
<li>clip_tp -- applies to SoftDiceScore. If TP cound is 0, then we clip it to a minimum value so it is not 0.</li>
<li>weight_dice -- applies to SoftDiceScore. Weight to which SoftDiceScore loss is multipled when calculating it.</li>
<li>weight_ce -- applies to CrossEntropy. Weight to which CrossEntrop loss is multipled when calculating it.</li>
<li>ignore_label -- applies to both SoftDiceScore and CrossEntropy. This is the label which is ignored during the loss calculations. Typically it is set to the value which is used for 0 padding during dataset creation.</li>
</ul> 
<li>LossConfigDeepSupervision -- configuration for loss which is actually a wrapper around some other loss function. This just applies loss on multiple different scales of auxiliary loss and sum them. It has three arguments:
<ul>
<li>loss_name --  name of the loss so the training script knows which loss to cast. It must be a unique name.</li>
<li>loss_function -- class instance of the valid/implemented loss function</li>
<li>cascade_number -- number of cascades in SNNUnet model. The real number of aux losses will be this-1 (the smallest resolution is skipped same way as it is skipped in NN-Unet implementation. Why... idk.)</li>
</ul>
</li>
</ul>

Each of the configs has traditional <b>methods</b>:

<ul>
<li><b>"__str__"</b> -- to print confing. Classes are printed as names.</li>
<li><b>"export_to_json"</b> -- method which accepts <b>export_path</b> as an argument which is a path to the json file where the confing will be saved. DISCLAIMER: one more time, propertis which are either class or function are stored as the the names of the classes and functions.</li>
<li><b>"import_from_json"</b>-- method which accepts <b>import_path</b> as an argument which is a path to the json file where the confing is stored. DISCLAIMER: one more time, propertis which are either class or function are stored as the the names of the classes and functions.</li>
</ul>

</details>

---

### soft_dice_score.py script

<details>
The script <span style="color:green;">soft_dice_score.py</span> is located at <span style="color:green;">CustomLosses/</span> directory and it implements the class <b>SoftDiceLoss</b> which is in charge of implemented soft dice loss. A good overview of the motivation behind soft dice loss can be found on following <a href=https://www.jeremyjordan.me/semantic-segmentation/>link</a>. However, this implementation offers a bit more fine tunning via its parameters:

<ul>
<li>apply_nonlin -- Callable -- if for some reason we want to provide probabilities, we can provide a function here 
which applies probability such as Sigmoid or Softmax. For instance, for compound loss it is: torch.softmax(x, 1). The input will be passed through the function provided here as callable.</li>
<li>batch_dice -- boolean -- if set to True the whole batch will be treated as a single image, not as separated cases coming
from different samples. If False, dice score will be caulcualte per sample in batch. Calculating on the batch level might 
smooth the variance in small datasets. In the nnUnet (default_experiment_planner.py) this are the values: 3Dlow_res = False, 
3Dful_res = True (in case of cascade), otherwise it is False. 2D = True. So in our case (fullres3D variant no cascade) this
param is set to False.</li>
<li>do_bg -- boolean -- If set to true, dice loss will include the background into calculation. Otherwise, it will calculate
loss only based on foreground pixels (casses). Default: False</li>
<li>smooth -- float -- factor to avoid division by 0. Default 1.0 but in nnUnet it was passed as 1e-5.</li>
<li>clip_tp --  float -- This can help to prevent minimum TP being 0 in some cases of highly unbalanced data.
Default: not provided so it is None.</li>
<li>ignore_label -- int -- label which is going to be ignore during the calculation of the loss.</li>
</ul>

How it works? Well, after intialisation of softmax via <b>\_\_init\_\_</b> method. The init is simple and just make all the properties avalable through the class. Method which is in charge of caluclation is named <b>forward</b> and it accepts two arguments: ground truth of the smaples (<b>y</b>) and models predictions for the samples (<b>x</b>). Then, based on the the batch_dice parameter it determines on which axes the calculation should be done (across the samples or each sample individually). This helps with variance smoothing. This is followed by applying nonlineariy given in <b>apply_nonlin</b> parameter and caluclating ignore_mask based on the <b>ignore_label</b> parameter. Then the necessary values <b>tp, fp, fn</b> for dice score caclulations are obtaine with help of the <b>__get_tp_fp_fn_tn</b> method which is pritty much self explainatory - it returns the values which are summed accross the given axes (for all samples together or per sample basis -- axis 0). Lastly, the dice loss (dice score) is caluclated with special care taken for avoiding 0s (either as a TP value <b>clip_tp</b> parameter or by <b>smooth</b> value to avoid division by 0). Lastly, the mean accross all classes is (even backgorund if paramterer <b>do_bg</b> is set to True) calculated.

</details>

---

### compound_loss_DC_and_CE.py script

<details>
The script <span style="color:green;">compound_loss_DC_and_CE.py</span> is located at <span style="color:green;">CustomLosses/</span> directory and it implements the class <b>DC_and_CE_loss</b> which is implementing the compound loss build from two other losses:

* soft_dice_loss -- as explained in this readme.
* CrossEntropyLoss -- simple cross entropy loss obtained from torch.nn (<a href=https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>link</a>).

How it works? Well theses two losses are combined together in the <b>forward</b> method by the following equation: 

$Loss = weight\_dice \cdot dice\_loss + weight\_ce \cdot ce\_loss$

where <b>weight_dice</b> and <b>weight_ce</b> are hyperparameters provided by the confing file to the <b>\_\_init\_\_</b>, while <b>dice\_loss</b> and <b>ce\_loss</b> are the losses calculated by the soft_dice_loss and cross_entropy_loss. The weights are typically set to the value 1.0.
</details>

---

### deep_supervision_loss.py script

<details>
The script <span style="color:green;">deep_supervision_loss.py</span> is located at <span style="color:green;">CustomLosses/</span> directory and it implements the class <b>DeepSupervisionLoss</b> which is implementing the deepsupervision loss. Deepsupervision loss is a loss wrapper which wraps arround a loss function enabeling combination of auxiliary losses. 

<br>

How it is done? Based on the provided names in the <b>\_\_init\_\_</b> method of the class the determine which loss to use and caluclates weights based on the number of cascades (given by <b>cascade_number</b> which can be obtained from the model config). Weights are caluclated as follows

$W_i = \frac{\frac{1}{2^i}}{\sum_i^\text{cascade\_number} W_i}; i \in [1,2,...,cascade\_number]$

The bigest weight(normalized) is associated to the segmentation map with the biggest resolution, where every consequtive one is having less and less importance (smaller weight). To be precise, the deep_supervision_loss is caluclated as follows (based on calulcated weights):

$Loss = \sum_i^\text{cascade\_number} W_i \cdot L_i$

where $L_i$ is loss at the i-th cascade/auxliliary loss output. 

</details>


### inspect_loss.py script

<details>
The script <span style="color:green;">compound_loinspect_lossss_DC_and_CE.py</span> is located at <span style="color:green;">CustomLosses/</span> serves the purpose to inspect and test losses. This is script is <b>not called or executed anywhere and serves the purpose of debuging and inspecting</b>. It can be invoked by calling <span style="color:green;">inspect_loss/</span> from cli and as an argument it take <b>export_path</b> which is a path to the export dir where the loss config files will be exported and later on imported to show the functionality of the script. 

</details>