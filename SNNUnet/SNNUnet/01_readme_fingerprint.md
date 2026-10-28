# Readme file for the Simplified version of the NN-Unet 


## Dataset fingerprint

To start to work with Simplified NN-Unet first we need to meet our enemy - and that is the data. Fingerprinting is a "fancy" way of saying dataset statistics. All in all, the fingeprtinting of the data is done with the scripts which can be find in ${\color{green}\text{"DatasetFingerprint/"}}$. Scripts are dealing with following tasks:

  * <b>Dataset splitting</b> - this part splits dataset into subsets: training, validation and test
  * <b>Dataset statistics == dataset fingerprinting</b> - this part creates datasets statistics and export it to the ".json" format. It is really important do obtain dataset statistics because based on this statistics the model is build and it highly influence the whole training process.


Scripts implemented in the fingerprinting are:

* <details> <hr> <summary> split_data_from_xlsx </summary> 
    Script which divides dataset into training, valid and test set based on the provided ".xlsx" file. Input xlsx file must contain the following columns :

    ```
    Image                   Segmentation                  Set
    <image name>.nii.gz     <segmentation name>.nii.gz    train/valid/test
    ```
    The xlsx file is located in the same directory which contains two subfolders: <i>img</i> and <i>seg</i> :

    ```
    some_dir:
        --> img
            --> <image name>.nii.gz
        --> seg
            --> <segmentation name>.nii.gz
        --> <name>.xlsx
    ```

    The script will based on the tag in "Set" <b>copy</b> files to the provided export dir path where it wll create following structure:

    ```
    export_dir_path:
        training_data:
                 --> img
                     --> <image name>.nii.gz
                 --> seg
                     --> <segmentation name>.nii.gz
        
        validation_data:
                 --> img
                     --> <image name>.nii.gz
                 --> seg
                     --> <segmentation name>.nii.gz
        test_data:
                 --> img
                     --> <image name>.nii.gz
                 --> seg
                     --> <segmentation name>.nii.gz
    ```


    <b>Args</b>:
    <ul>
    <li><b>input_xlsx_path</b>, string,  path to the "xlsx" file which is placed in the directory which also contains the 
    img and seg directories.</li>
    <li><b>export_dir_path</b>, string, path to the location where the export dirs with the structure explained above will be
    created. </li>
    </ul>

    <b>Output</b>:
    <ul>
    <li><b>organized directories and files</b> - as explained above.
    </ul>
   
    Script which implements it is named: ${\color{green}\text{"split\_data\_from\_xlsx.py"}}$
  </details>

* <details> <hr> <summary> split_data_from_ratio </summary> 
  Script which divides dataset into training, valid and test set based on the provided ratio. For given ratio E.G. 75% it means that randomly selected 75% files from the input directory will go to training set. The other 25% will be randomly split to 12.5% and 12.5% where first 12.5% data represents validation dataset and other 12.5% represents test set. 

  The scripts expects that input dir has the following structure:

  ```
  some_dir:
      --> img
          --> <image name>.nii.gz
      --> seg
          --> <segmentation name>.nii.gz
  ```
  
  Scripts seprates files in three directores which are generated in the output dir which is created from scratch (it overrides the dir so it will DELETE all the content). The files from the input dir are copied to the output dir. The structure of the output dir is:

  ```
  output_dir:
      training_data:
                --> img
                    --> <image name>.nii.gz
                --> seg
                    --> <segmentation name>.nii.gz
      
      validation_data:
                --> img
                    --> <image name>.nii.gz
                --> seg
                    --> <segmentation name>.nii.gz
      test_data:
                --> img
                    --> <image name>.nii.gz
                --> seg
                    --> <segmentation name>.nii.gz
  ```
  <b>Args</b>:
  <ul>
  <li><b>input_dir</b>, string, path to the input dir which have two subdirs: img and seg</li>
  <li><b>output_dir</b>, string, path to the location where the export dirs with the structure explained above will be
  created. </li>
   <li><b>ratio</b>, float, ratio of the data which is kept in the training set. The ramainder is equaily split between
   validation and test set. It must be less than 1.0.</li>
   <li><b>seed</b>, int, since the data is randomly split, to control randomness and repeatability of the experiments, 
   it is possible to set a random seed.</li>
  </ul>

  <b>Output</b>:
  <ul>
  <li><b>organized directories and files</b> - as explained above.
  </ul>
  
  Script which implements it is named: ${\color{green}\text{"split\_data\_from\_ratio.py"}}$
  </details>


* <details> <hr> <summary> generate_dataset_fingerprint </summary> 
  This script handels calculates statistics of the dataset and is based on the class <b>DatasetFingerprint</b> which handels all the export. It expect to have the following structure of directorires/files (for instance it can/should be <i>training_data/img</i> and <i>training_data/seg</i>):

  ```
  CT_Volumes_dir:
      --> ct__001.nii.gz
      --> ct__002.nii.gz
      --> ...
      --> ...
      --> ...
      --> ct__n.nii.gz
  
  CT_segmentations_dir:
      --> ct_seg_001.nii.gz
      --> ct_seg_002.nii.gz
      --> ...
      --> ...
      --> ...
      --> ct_seg_n.nii.gz
  labels.json
  ```

  Of course, the naming does not need to be exact, but there must be correspondance between files in volumes dir and segmentation dir.
  File <i>labels.json</i> contains labels. The file must comply to the following <b>dictionary</b> structure:
  
  ```
    {"labels":{
    "0": "background",
    "1": "label#1",
    "2": "label#2",
    ...
    ...
    ...
    "n": "label#N"
    }
    }
  ```

  About what is dataset fingerprint and how to use it, it will be more said in the following chapters of this readme file since the is only the script which starts the whole process.


  <b>Args</b>:
  <ul>
  <li><b>volume_dir_path</b>, string, path to the directory containing CT volumes (nii.gz) based on which statistics will be caluclated </li>
  <li><b>segmentations_dir_paths</b>, string, path to the location where segmentation maps are loacated (nii.gz) based on which label
  statistics is calculated. DISCLAIMER: THE VOLUME AND SEGMENTATIONS FILES MUST BE CORRESPOINDING TO EACH OTHER.</li>
   <li><b>labels_mapping_dict_path</b>, string, path to the label mapping dict which content is presented above.</li>
   <li><b>export_path</b>, string, path to the file (.json) where the datastatistics will be stored.</li>
   <li><b>number_of_processes</b>, int, if lest empty, None will be used which means no multiprocessing. If an integer is provided, then that many processes will be spawn.</li>
  </ul>

  <b>Output</b>:
  <ul>
  <li><b>.json file</b> - a .json file which contains all dataset statistics in a nicely formated form. This is I believe important
  because dataset statistics is often needed for papers/presentation - so I have put a bit of additional effort to make it look nice.
  </ul>
  
  Script which implements it is named: ${\color{green}\text{"split\_generate\_dataset\_fingerprint.py"}}$.
  <hr>
  </details>



<hr>

## DataFingerprint class

Here I want to present logic behind <b>DataFingerprint</b> class implemented in ${\color{green}\text{"DatasetFingerprint/Utils/utility\_functions.py"}}$. To better understand which statistic do we meassure and calculate here is the overview of it:

* <details> <hr> <summary> GENERAL_INFO </summary> 
  <ul>
    <li><b>"Number_of_samples"</b> -- how many samples are in the given folders for which the statistics is being made.</li>
    <li><b>"Median_axis_size"</b> -- median size per axis across all given samples.</li>
    <li><b>"Mean_axis_size"</b> -- mean size per axis across all given samples.</li>
    <li><b>"Median_axis_size_cropped"</b> -- We crop the images only to their useful content (volume map != 0 which is background). This is the median size of it.</li>
    <li><b>"Median_spacing_value"</b> -- Median spacing value is a median of all spacings per axis.</li>
    <li><b>"Mean_spacing_value"</b> -- Mean spacing value is a mean of all spacings per axis.</li>
    <li><b>"Median_fg_to_bg_ratio"</b> -- Number which represents the median ratio of foreground pixels (voxels belonging to some class) to the whole volume.</li>
    <li><b>"Mean_fg_to_bg_ratio"</b> -- Number which represents the mean ratio of foreground pixels (voxels belonging to some class) to the whole volume.</li>
    <li><b>"Mean_number_of_channels"</b> -- Although it is shown as a float, this should always be a whole number because it represents the mean number of channels across the given samples. If it is not a whole number, something is not right.</li>
    <li><b>"Mean_z_channel_order"</b> -- Same as the mean number of channels: this is a float but it must be a whole number. It is important because all samples must be consistent having the same z channel order and the same number of channels. The program will not fail, but it is a problem which the user must solve.</li>
  </ul>

  <hr>
  </details>

* <details> <hr> <summary> LABELS_MAPPING </summary> 
  This is a nice reminder of mapping from number to labels. It is also important to notice that beside default background
  label with <b>reserved int value of 0</b>, there is added <b>ignore label</b> with value of <b>-1</b> which is a label used when we
  add something to the image. E.G. we rotate the image and needs to 0 pad it, we 0 pad the volume, but in segmentation map we put -1 so we know that we have 0 pad it and that those added pixels are not part of the original map.
  <hr>
</details>

* <details> <hr> <summary> VOXEL_STATISTICS_SAMPLES </summary> 
  To calculate this statistics the following steps are perfomed:
  
  1. For each sample we randomly sample 10000 forground voxels. 
  2. Those samples are concatenated to one list.
  3. Statistics of that list is caluclated.

  The statistics contains: <b>min</b> (minumum value), <b>max</b> (maximum value), <b>mean</b> (mean of the all sampled voxels), <b>std</b> (standard deviation of sampled voxels), <b>precentile 0.05</b>, <b>median</b>, and <b>percentile 99.5</b> (representes different percentiles of sampled voxels)
  <hr>
</details>

* <details> <hr> <summary> VOXEL_STATISTICS_MEANS </summary> 
  This statitics, althought might seem same like the VOXEL_STATISTICS_SAMPLES, it is calculated differently:
  
  1. For each sample we extract all foreground voxels.
  2. We calculate the same statistics for these voxels as in VOXEL_STATISTICS_SAMPLES: <b>min</b> (minumum value), <b>max</b> (maximum value), <b>mean</b> (mean of the all foreground voxels), <b>std</b> (standard deviation of foreground voxels), <b>precentile 0.05</b>, <b>median</b>, and <b>percentile 99.5</b> (representes different percentiles of all foreground voxels)
  3. For each case now there exist dictionary of its statistics. The <b>mean</b> values of all dictionaries/statistics is calculated and saved as final statistics. It can be seen as "mean of mean".
  
  Again, the final statistics contains: <b>min</b> (minumum value), <b>max</b> (maximum value), <b>mean</b> (mean of list which contains mean values of samples), <b>std</b> (mean of list contatining all std of samples), <b>precentile 0.05</b>, <b>median</b>, and <b>percentile 99.5</b> (representes mean of list contatining different percentiles of cases)
  <hr>
</details>

* <details> <hr> <summary> CLASS_STATISTICS </summary> 
  Finally, class statistics provides for each class statistics which gives better insights in distribution and size of every class in the dataset. Here are the metrics measured on class basis:
  <ul>
    <li><b>"Number_of_samples"</b> -- Class occures in this number of samples. It can be corelated with total number of samples given in <i>GENERAL INFO -- Number_of_samples</i></li>
    <li><b>"Median_size"</b> -- For each sample which contain observed class the number of voxels is calculated. This number
    representes median number of voxels belonging to this class.</li>
    <li><b>"Mean_size"</b> -- For each sample which contain observed class the number of voxels is calculated. This number representes median number of voxels belonging to this class.</li>
    <li><b>"Median_class_to_vol_ratio"</b> -- This is median of all ratios calculated as number of class voxels against all voxels in observed sample.</li>
    <li><b>"Mean_class_to_vol_ratio"</b> -- This is mean of all ratios calculated as number of class voxels against all voxels in observed sample.</li>
    <li><b>"Median_class_to_fg_ratio"</b> --  This is median of all ratios calculated as number of class voxels against all foreground voxels in observed sample.</li>
    <li><b>"Mean_class_to_fg_ratio"</b> -- This is mean of all ratios calculated as number of class voxels against all foreground voxels in observed sample.</li>
  </ul>
  <hr>
  </details>


### Implementation

As mentioned, dataset fingerprint is generated in script ${\color{green}\text{"DatasetFingerprint/Utils/utility\_functions.py"}}$ where two main block can be seen.

1. <b>read_data_fingerprint_json</b> -- function which accepts path to the fingerprint file (output of generate_dataset_fingerprint script) and returns the dictionary as result. Because ".json" is nicely formated, this function is needed whenever we want to read json file. I know it is annoying, but I really like the format. It can be easily avoided with some small code modifications.

2. <b>DataFingerprint</b> -- class which enables generation of all statistics regrading the dataset. I will briefly present methods and their mechanisms here, but I have also documented the class:
  
    <ul>
      <li><b>__init__</b> -- init function which generates three important lists: <i>self.CT_files</i> (contain files of images), <i>self.seg_maps</i> (contain segmentations), and <i>self.labels</i> (contain labels like label names, this is a dictionary). This files are populated thorugh methods <b>__obtain_files</b> and  <b>__obtain_labels</b></li>
      <li><b>"__obtain_files"</b> -- Method which lists the files in path_to_files and extract all files which ends with desired file_extension. Basicaly list of files. Also it sorts them so it is always consistant (it ensures compatibility between segmentations and volumes.)</li>
      <li><b>"__obtain_labels"</b> -- Method which lists the labels in the given json file presented as path_to_labels. Result is a dict with where key is voxel value, while value is labels' name (LABELS_MAPPING)</li>
      <li><b>"export_fingerprint"</b> -- <b>MAIN METHOD</b> and entery point. This method calls all other methods and generates statistics. Also it exports generated statistics to a file (json). It obtain all necessary data through <b>__obtain_axis_sizes</b> and <b>__obtain_class_properties</b> methods</li>
      <li><b>"__obtain_axis_sizes"</b> -- Method which obtains dataset properties such as sizes, medians, fg/bg ratios etc (populates VOXEL_STATISTICS_SAMPLES and VOXEL_STATISTICS_MEANS). 
      It support multiprocessing.</li>
    </ul>

