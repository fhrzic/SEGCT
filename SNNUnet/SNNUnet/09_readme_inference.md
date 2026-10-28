# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Inference"}}$ directory

As it can be seen, the directory <span style="color:green;"> "Inference"</span> the following scripts can be found:

```
    Training:
        --> inference_utility_functions.py
        --> inference_container.py
        --> perform_inference.py
        --> inference_statistics_evaluation.py
        --> run_inference_from_dir.py
        --> archive_model.py
```

The scripts contents/purposes are as follows:

<ul>
<li>"inference_utility_functions" -- all utility functions used to do inference. The idea is that this script holds a handfull of useful function which can be used as building blocks for various inference schemes. If someone wants to build its own inference, these functions will come in handy. </li>
<hr>
<li>"inference_container" -- implements <b>InferenceContainer</b> class which is the cornerstone class of the implemented inference. It is used for prediction and returing the results.</li>
<hr>
<li>"perform_inference" -- main script for inference. It is used to conduct inference on directory containing cases or a single directory. With this script one can test if the inference is working. It is also based on its own <b>configuration class</b> implemented in <span style="color:green;"> "confing_inference.py"</span> script.</li>
<hr>
<li>"inference_statistics_evaluation" -- script which is called on top of perform_inference to obtain quantitative evaluation of cases. This script produces the final statistical report, while <b>perform_inference</b> generates model's predictions.</li>
<hr>
<li>"run_inference_from_dir" -- script  which runs inference utilizing models which can be find in the <span style="color:green;"> "TrainedModelsArchive"</span> directory. This is usefull when dealing with containers!</li>
<li>"archive_model" -- script  script which archives/saves model (minimum context) from the models output directory to <span style="color:green;"> "TrainedModelsArchive"</span> directory. This is usefull when dealing with containers and wanting to perform inference on a different PC.</li>


</ul>

Callable scripts inteded to be called via comand propt are:

<ul>
<li><b>perform_inference</b> -- script which performs inference on directory or single file.</li>
<li><b>inference_statistics_evaluation</b> -- script which caluclates statistics once when <b>perfrom_inference</b> has been called.</li>
<li><b>run_inference_from_dir</b> -- script which run inference based on the models located in  TrainedModelsArchive directory.</li>
<li><b>archive_model</b> -- script which saves model to <b>TrainedModelsArchive</b> directory for easier protability.</li>
</ul>

---

## inference_utility_functions.py script
<details>
Script <span style="color:green;"> "inference_utility_functions.py"</span> located inside directory <span style="color:green;">Inference</span> directory implements <b>A LOT OF FUNCTIONS</b> usefull for data transformations during the inference. Here is the list and brief explanation of function provided in this script

<ul>
<li><b>"symetrical_padding":</b>
<details>
Function which padds the input <b>volume</b> to the <b>desired_shape</b>. If the input volume is bigger than desired shape (in all dimensions), then there is nothing to be padded in the first phase of padding. If some or all dimensions are smaller than the desired_shape, then the volume is padded symetrically in that dimensions with <b>padding_value</b>. In second phase of padding it is possible to force the padded volume to be  divisible by the provided <b>patch_shape</b>. The need for second phase of padding is controled by <b>divisible_by_patch</b> boolean varaible. The idea is to make thinks easier and make experimenting with various input sizes possible. The function returns padded tensor and padding offset along with original volume shape (the latter two are needed for reconstruction of the volume).
</details>
</li>
<hr>
<li><b>"unpad_volume":</b>
<details>
Function which "cancels the padding". As an argument it accepts three params needed to reconstruct the original volume which can be obtained from <b>symetrical_padding</b> funtion: <b>volume</b>, <b>padding_offset</b>, <b>original_volume_shape</b>. Based on these parameters, a region which starts in <b>padding_offset</b> coordinates and is of size <b>original_volume_shape</b> is cropped and returend from the provided <b>volume</b>.
</details>
</li>
<hr>
<li><b>"split_volume_into_patches_no_padding":</b>
<details>
This function splits input <b>volume</b> into nonoverlapping patches of <b>patch_shape</b> unless the parameter <b>overlap</b> is set to a value, E.G. 0.5 means that every patch will have 50% of the previous patch also. The overlaping is created by sride in a fashion of a moving window. The only thing is the residuals: for instance the <b>volume</b> is not divisible by <b>patch_shape</b> which results in some "uncovered" area. This patches are by createing patch only over the uncovered area: E.G. 1D example is that "volume" has lenght of 130 dots and patch is 50 with overlaping of 0.5. First patch is 0-50, second is 25-75, third is 50-100, fourth is 75-125, fifth would be 100-150 which is not possible, so we squeeze it in: 70-130. Here is no zero padding involved! It returns patches as staked torch tensor and patches location to allowed reconstruction of the volume (beggining coordinates of every patch).
</details>
</li>
<hr>
<li><b>"reconstruct_volume_from_patches":</b>
<details>
This function is nightmare due to the out-of-memory errros. So to avoid them there is a lot of tensor moving between CPU and GPU. Banchmarking showed that from all the components - moving tensors from one device to another is the biggest bottleneck of the whole inference pipeline. How this method works... First it creates and empty tensor where are the predictions will be stored (dimension: <b>(number_of_classes, volume_shape)</b>). Then it calculates gaussian kernel by calling <b>compute_gaussian</b> method which returns pondering/weightening kernel if the flag <b>gaussian</b> is set to True. Each patch is multiplied by this kernel in order to reduce influence of border predictions on the patch. Then it iterates over patches and multiples them with gaussian while saving the predictions on the right locations in the created empty storage tensor. Then the counting of overlaps at each location is done which is necessary number for normalization. Normalization is done by dividing predictions on each loacation by the number of overlaps on that location. At the end, the reconstructed volume (empty tensor created at the beggining of the function is now populated with predictions) is returned. 
</details>
</li>
<hr>
<li><b>"compute_gaussian"</b>
<details>
Simple function which create a gaussian kernel over the provided <b>patch_scale</b> with the peak in the center of it. This is necessary for weightening the focus of the prediction on the central of the patch and not on its borders.
</details>
</li>
<hr>
<li><b>"transfer_ct_to_npy":</b>
<details>
As the name of the function says: it transfer a nifti file found in the provided <b>path_to_ct</b> to the numpy array (which can be easily casted to torch tensor) while performing the normalization provided in <b>normalization_fingerprint_path</b> variable. The function loads volume with help of <b>"sitk"</b> library, transer it to numpy, rearrange the channels and applies nromalization if normalization_fingerprint_path is provided. If it is set to <b>None</b>, no normalization is done!
</details>
</li>
<hr>

<li><b>"transfer_to_sitkImg":</b>
<details>
Function which exports numpy volume to <b>"sitk"</b> image. It requires <b>data</b> which is numpy array that needs to be transfered to sitk format, and <b>original_ct_path</b> which contains all necessary propertis to make transfer possible (such as> origin, spacing and direction).
</details>
</li>
<hr>

<li><b>"obtain_statistics":</b>
<details>
Function which based on privded <b>gt_segmentation_path</b> and <b>pred_segmentation_path</b> calculates following metrics: "tp", "fp", "fn", "precision", "recall", "dice", "f1_score". The export is done in the provided <b>export_path</b> in ".json" format. Input can be both numpy array ".npy" or "nifti" files. 
</details>
</li>
<hr>

<li><b>"plot_metrics":</b>
<details>
Function which plots metrics provided in <b>metrics_avg</b> which are the averaged metrics obtained in script <b>merge_statistics</b>. Metrics are: "precision", "recall", "f1_score", and "dice". Honestly, see merge_statistics because this is highly connected to that script for exact metrics and their meaning. The end result is shwon and exported in provided <b>output_path</b> under the name "merged_metrics.png".
</details>
</li>
<hr>

<li><b>"merge_statistics":</b>
<details>
This function is tightly connected with <b>obtain_statistics</b> function. Obtain_statistics calculate individual statistics and store them in the export dir. Merge_statistics caluclate and merges statistics for all individual statistics by <b>averaging</b> them on per-metrics basis. Then it exports them in <b>output_path</b> directory under the name <b>"merged_metrics.json"</b>. The script scans for all ".json" files in the provided <b>input_path</b> because it assumes that the inference will be done by <b>perform_inference</b> scirpt which in the output dir has only two types of ".json" files: "times.json" and "statistics_jsons" files(individual statistics for each file). 
</details>
</li>
<hr>

<li><b>"plot_times":</b>
<details>
This function plot times provided by <b>times_json_path</b> which points to "times.json" created by <b>perform_inference</b> script. It also have <b>output_path</b> to the directory where image <b>"times.png"</b> will be stored. Times represent time needed to do a single predict on the given case.
</details>
</li>
<hr>
</ul>
</details>
</details>


## inference_container.py script
<details>
Script <span style="color:green;"> "inference_container.py"</span> located inside directory <span style="color:green;">Infernece</span> implements <span style="color:green;">InferenceContainer</span> class. This class hides "all" complexity of inference and it is supposed to make it easy. Here is the list of all methods which are implemented in the class:

<ul>
<li><b>"__init__"</b>:
<details>
Initialisation method which accepts as input <b>inference_config</b>, instance of <b>InferenceConfig</b> implemented in script  <span style="color:green;">config_inference.py</span> located in  <span style="color:green;">Configurations</span>. This is to be consistant with previous logic where everything is conected via configuration classes. Init method first obrain number of classes through <b>__obtain_number_of_classes</b> method, and then obtains model via <b>__obtain_model</b> method. Model is obtained based on the provided <b>model_path</b> in configuration and is set to the <b>device</b> (also provide in inference_config). After it, it creates export directory if it does not exists provided by <b>output_dir</b>. The natural behaviour of the inference is to append to the export dir, just in case that somebody wants to do targeted inference of cases and have everything on one place. If <b>filp</b> is set to true in configuration, this method alsi obtain axes through <b>__obtain_mirror_axes</b> method and, finally, it obtain patch size through method <b>__obtain_patch_size</b>.
</details>
</li>

<li><b>"__obtain_number_of_classes"</b>:
<details>
Simple method which returns number of classes in the dataset. This information is necessary for reconstruction of the volume from patches (inference is done on patches). The number of classes is estimated from <b>data_fingerprint</b> propert of the provided <b>inference_config</b>.
</details>
</li>

<li><b>"__obtain_model"</b>:
<details>
Method which loads model from the porvided <b>model_path</b> and places it on specified <b>device</b>. 
</details>
</li>

<li><b>"__obtain_patch_shape"</b>:
<details>
Method which based on the provided <b>model_config_path</b> loads the patch size on which the model was trained. Now, if for some reason we want to change inference input patch, that is posible by providing a tuple instead of a path (E.G. (96,96,112)). However, model must be flexible to accept any size. 
</details>
</li>

<li><b>"__obtain_mirror_axes"</b>:
<details>
Method which obtain mirror axes dimension. The idea is that if the patch is mirrored accross all axes and then the predictions on the differently oreineted patches are agregated, the final outcome will be more robust. Hence, this method provides list of axes to be mirrored in multiple combiations.
</details>
</li>

<li><b>"__obtain_background_label"</b>:
<details>
Method which from provided <b>data_fingerprint</b> obtains the backgorund label (it reads fingerprint from the provided data_fingerprint). This is importat for zero-padding during the model prediction, specially in cases where the patch size is bigger than the provided volume.  
</details>
</li>

<li><b>"__model_predict"</b>:
<details>
Method which implements a small pipline necessary to obtain predictions form the given input volume. It returns segmentation for the provided volume. The pipline is as follows:

<ol>
<li>The model is padded by calling <b>symetrical_padding</b> function implemented in <b>inference_utlity_functions</b>. Result is volume which is divisible by the provided patch shape.</li>
<li>Then the model is splitted with help of <b>split_volum_into_patches_no_padding</b> function into patches. The volume is splitted into patches which may or may not overlap(parameter <b>overlap</b>). The patch shape is obain in the init method. </li>
<li>All the patches are stacked in one big tensor. The script now split that huge tenosr into groups == batches so the prediction can be carried on in a more efficient way. Remember, the slowest part of prediction is data transfer.</li>
<li>Now it obtain predictions for all patches in a batch including possible mirrored versions of the patches. They are aggreated and averaged. At the end, for each patch there is one prediction which is added to a storage named <b>_tmp_predictions</b>.</li>
<li>Now is time for reconstruction, the volume is reconstructed from patches with help of <b>reconstruct_volume_from_patches</b> function  implemented in <b>inference_utility_functions</b>. It acceptes a lot of parameters from which maybe the most important is <b>gaussian</b> which specifies if the predictions will be pondered by gaussian kernel. This is important in case of the overlap because then, when i.e. two patches overlap, it is not same if a voxel was at the end of one patch or in the middle of another patch. If it is in the middle the model had much bigger context and the prediction is more truthworthy than in the case if the same voxel was in the i.e. corner of another patch where the context is quite limited. How is this done? Easy: we create empty--zero matrix of same shape as the input volume. It is known on which position is every patch. Prediction of patch is multiplied by gaussian kernel and added to its position. The final volume will for each position have summed all predictions on the patches that involved that particular position. All positions at the end are averaged by dividing them with the number of patches that involved that position.  </li>
<li>So, after reconsturcting the volume, it is still needed to find to which class the prediction belongs. This is done by argmax across the channel dimension because each channel representes one class. </li>
<li>Lastly, the predicted segmentation map is unpaded by the function <b>unpad_volume</b> from <b>inference_utility_functions</b>. This is necessary so the predicted segementation can be overlayed over the original volume in case that is viewed in slicer.</li>
</ol>

</details>
</li>
<li><b>"obtain_prediction"</b>:
<details>
This method is <span style="color:green;"><b>ENTRY POINT</b></span> for inference. It contains couple of arguments mostly defining what will be save in the output dir. The mandatory argument is <b>path_to_src</b> which is path to the source file of the volume for which we want to make a prediction, wheter it is "nifti" or already normalized "numpy" array (".npy). Next argument is not used, but it is convinient to copy ground truth segmentation to the same file, specially when calculating statistics. Hence, if <b>path_to_segmentation</b>
is provided, it will copy the segmentation from this path to the case output dir (preserving its format--extension). Same is with the input volume provided on <b>path_to_src</b> which if flag in <b>save_original</b> is set to true, will be copied to the target location. Lastly if flag <b>save_npy</b> is set to true, the numpys versions will also be exported. Now, there is a lot of things that can be exported so it can easily create a lot of data. Also exporting mulitple things slowes down the process of inference.
</details>
</ul>
</details>

---

## perform_inference.py script
<details>
Script <span style="color:green;"> "perform_inference.py"</span> located inside directory <span style="color:green;">Inference</span> is main script to be called when performing the inference. This script utilises the above created scripts. It is the entry point for inference and can be invoked by <b>comand line</b>.

How does it work? Well it accepts a lot of arguments, build instance of <b>InferenceContainer</b> from <b>inference_container.py</b> and uses it to predicts the output segmentation maps. For each provided case for which prediction is required, in the export dir it creates a directory which will contain everything wanted for that case(rembember: <b>obtain_prediction</b> method offers to export volume, segmentations, numpys...). In the same output directory script will create <b>"times.json"</b> file which contains prediction times for each case and "inference_config.json" which contains inference params just for the recording purposes. This is how the script operates:
<ol>
<li>First it build the InferenceConfig instance used to build the inference container.</li>
<li>Then it creates inference container which is used to obtain predictions.</li>
<li>Now it scans for the files: if directory is provided as the input, it lists all the files in it.</li>
<li>Prediction is happening> iterate over all files and obtain/export predictions!</li>
<li>Export times and inference-config files to <b>"times.json"</b> and <b>"inference_config.json"</b> files</li>
</ol>

</details>

---

## inference_config.py script
<details>
Script <span style="color:green;"> "config_inference.py"</span> located inside directory <span style="color:green;">Configurations</span> implements <span style="color:green;">InferenceConfig</span> class. It contains all the important attributes for conducting inference

<b>Attributes:</b>
<ul>
    <li><b>"model_path"</b> -- string -- path to model weights which will be used for prediction.</li>
    <li><b>"model_config"</b> -- string -- path to model config which will be used for prediction.</li>
    <li><b>"output_dir"</b> -- integer -- directory in which the outputs/predictions will be stored.</li>
    <li><b>"device"</b> -- string -- device on which the inference will be performed. E.G. "cuda:0".</li>
    <li><b>"verbose"</b> --  boolean -- verbose flag to show prinitng.</li>
    <li><b>"batch_size"</b> -- int -- size of the batch used for inference.</li>
    <li><b>"normalization_fingerprint"</b> -- string -- path to normalization fingerprint.</li>
    <li><b>"data_fingerprint"</b> -- string -- path to dataset fingerprint.</li>   
    <li><b>"overlap"</b> -- float  -- percentage of overlap between patches (e.g 0.5 is 50% overlap).</li>
    <li><b>"gaussian"</b> -- bool -- if set to True, each patch will be multiplied with Gaussian kernel.</li>
    <li><b>"flip"</b> -- bloolean -- if set to True, each patch will be flipped on all axes.
</ul>

<br>

Many of the required attributes can be calculated or extrapolated from the dataset fingerprint file. That is why this implementation of 
<span style="color:green;">ModelConfig</span> class is also having init method. Here is the overview of the methods implemented in the class.

<b>Methods:</b>
<ul>
    <li><b>"__str__"</b> -- method which prints all the parameters stored in the inference confing.</li>
    <li><b>"export_to_json"</b> -- it exports all properties to the json file defined by the given argument <b>export_path</b>.</li>
    <li><b>"import_from_json"</b> -- it imports all properties from the json file defined by the given argument <b>import_path</b>.</li>
</ul>

</details>

## inference_statistics_evaluation.py script
<details>
Script <span style="color:green;"> "inference_statistics_evaluation.py"</span> located inside directory <span style="color:green;">ConfigInferenceurations</span> implements <span style="color:green;">InferenceConfig</span> class. It takes several arguments as the input and support couple scenarios based on them.

<b>Arguments</b>
<ul>
<li><b>path_to_gt</b> -- string -- path to directory or file where the ground truth segmentations are stored.</li>
<li><b>path_to_prediction</b> -- string -- path to directory or file where predicted segmentations are stored.</li>
<li><b>export_path</b> -- string -- path to the directory where statistics for each case will be stored along with merged statistics and time evaluation.</li>
<li><b>times_path</b> -- string -- path to <b>time.json</b>.</li>
<li><b>path_to_data_fingerprint</b> -- string -- path to the dataset fingerprint mainly used for label name extraction.</li>
<li><b>number_of_processes</b> -- integer -- number of processes to use for evaluation. If set to <b>None</b> single proces will be used.</li>
</ul>

So for each case (ground_truth and prediction) the script will create <b>"json"</b> file in <b>export_path</b> which contains statistics for each class as well as micro and macro overall performance. Once when this statistics is obtain for all files, <b>merged_metrics.json</b> is created along with <b>merged_metrics.png</b> which represents averages per class for all cases. If <b>times.json</b> is provided, then the same is done for the times also. At then end, there is a complete statistics provided in the export dir: both per case and overal (tp,fp,tn,fn,precision,recall,f1-score, macro and micro caluclation).

Now, there are three possible inputs for <b>path_to_gt</b> and <b>path_to_prediction</b>:

<ol>
<li><b>file-file</b> if two files are provided, then only this file is processed.</li>
<li><b>dir-dir</b> if same directory is provided for gt and predictions, then it is assumed that every subdir inside of it contains <b>original_segmentation.nii.gz</b> and  <b>predicted_segmentation.nii.gz</b> files. If <b>perform_inference</b> is used and <b>save_original</b> is set to True, this files will be generated.</li>
<li><b>dri1-dir2</b> in case that <b>perform_inference</b> is called without any modifications, it will export only <b>predicted_segmenation.nii.gz</b> in the respective case folder. Which means that ground truth segmetnations are located in another directory which needs to be provided.</li>
</ol>
</details>

## run_inference_from_dir.py script
<details>
Script <span style="color:green;"> "run_inference_from_dir.py"</span> located inside directory <span style="color:green;">Inference</span> implements method which strongly relies on models located in <b>TrainedModelsArchive</b> directory where they can be exported by utilizing command <b>archive_model</b>. This script is same as <b>perform_inference</b> but it requires less arguments and avoids absolute paths (usefull for container run!).

<b>Arguments:</b>
<ul>
    <li><b>"input_ct_dir"</b> -- string -- path to ct file(s)</li>
    <li><b>"config_dir_name"</b> -- string -- This is basically the models name</li>
    <li><b>"output_dir"</b> -- integer -- directory in which the outputs/predictions will be stored. They are stored in <span style="color:green;">TrainedModelsArchives/<i>config_dir_name</i>/Predictions/<i>output_dir</i></span>.</li>
    <li><b>"device"</b> -- string -- device on which the inference will be performed. E.G. "cuda:0".</li>
</ul>

This script is callable by cli function <b>run_inference_from_dir</b>! 

</details>

## archive_model.py script
<details>
Script <span style="color:green;"> "archive_model.py"</span> located inside directory <span style="color:green;">Inference</span> implements functionalty which transfers trained model's files to the <b>TrainedModelsArchive</b> directory located inside <b>SNNUnet</b> so it is easier to be run. The transfered files are:
"normalization_fingerprint.json", "training_fingerprint.json", all files in "Configs/*", and from "Checkpoints/*": "training_log.json", "last_checkpoint.pth", "best_model_checkpoint.pth", "best_model.pth"

<b>Arguments:</b>
<ul>
    <li><b>"input_dir"</b> -- string -- path to directroy which was used as an output to the model's training. The main directory which holds data and TRAINING dir.</li>
    <li><b>"name"</b> -- string -- name under which the model will be stored. This is the name of the dir which can be provided to <b>run_inference_from_dir</b> script to do the inference.</li>
</ul>

This script is callable by cli function <b>archive_model</b>! 

</details>
