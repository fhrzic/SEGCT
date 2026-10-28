# Readme file for the Simplified version of the NN-Unet 


## Dataset Preprocessing

The script to which this readme refere are implemented in: ${\color{green}\text{"DatasetPreprocessing/"}}$

Once when we have fingerprint of the data calculated, it is necessary to preprocess the data. Preprocessing generally means obtaining additional <b>per-case info</b>. Therefore, case information, as well as exporting NIFTI files to ".npy" format are the finals outputs of this step. Here is an short overview of what is being done in this step and why:

* <details> <hr> <summary> Cropping to usefull content </summary> 
    A stated in NN-UNet, some datasets might have a lot of "empty" border around the image. So all data is strip of that usless data.
    <p style="color:red;">TODO: recovery from crop (necessary data is stored in loc file)</p>
  <hr>
  </details>

* <details> <hr> <summary> Normalization </summary> 
    There are four normalization scheemes implemented, but they are all based on the Z-Score normalization. More info about Z-Score normalization can be found on the following <a href="https://www.statology.org/z-score-normalization/">link</a><href >. Basically, the four types distinguish one from another in way how the mean and std are computed.

  <hr>
  </details>
* <details> <hr> <summary> Export </summary> 
    Once the volumes are preprocessed/formated, they are exported to the ".npy" formats. Also, one additional file is exported which holds usefull proprety about foreground locations for each class.
  <hr>
  </details>



Two scripts are implemented in preprocessing:

* <details> <hr> <summary> dataset_formating</summary> 
    Main script for dataset formating. It loads volumes and segmentations, and export them to the ".npy" arrays while cropping the volumes and normalizing them. Segmentation maps are filtered only to valid values given in <i>fingerprint file</i> which is generated in <i>generate_dataset_fingerprint</i>. This script performs all above mentioned steps. The output of the script is file generated in desired location (<i>dir_to_export_path</i>) with the following structure:

    ```
    dir_to_export_path:
        ct:
            --> case_01.npy
            --> case_02.npy
            ...
            ...
            ...
            --> case_n.npy
        
        seg:
            --> case_01.npy
            --> case_02.npy
            ...
            ...
            ...
            --> case_n.npy     
        loc:
            --> case_01.npz
            --> case_01_surfaces.npz
            --> case_02.npz
            --> case_02_surfaces.npz
            ...
            ...
            ...
            --> case_n.npz    
            --> case_n_surfaces.npz
    normalization_fingerprint.json

    ```
    <b>IMPORTANT: Names of file does NOT need to be case_n, but they must have correspoinding names along the three directories!</b>
    
  <i>ct</i> holds cropped and normalized ct files, <i>seg</i> contains verified segmentation files, while <i>loc</i> contain <b>properties</b>. Properties are dictionaries where <b>key</b> is label present in the corresponding segmentation file located in seg folder under the same name, and <b>value</b> is list of locations (x,y,z) where the label is present in the corresponding segmentation file. Now, <b>"_surfaces"</b> extension contain only those voxels which are present at the surface of the segmentatation volume/blob. I have some idea to use this later on and currently it is not used anywhere.

    <b>Args</b>:
    <ul>
    <li><b>input_dir_path</b>, string,  path to the directory which has the following structure:
    
    ```
    input_dir_path:
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
    This structure is obtained by the <i>fingerprint</i> scripts, but it can also be created manually.
    </li>



    <li><b>normalization_fingerprint_path</b>, string, path to the fingerprint file which is used for labels and for some normalizations</li>
    <li><b>number_of_processes</b>, int, number of processes allocated for the dataset formating. It is usually 1 process per case. More processes-->faster export</li>
    <li><b>export_dir_path</b>, str, path to the export dir where the following structure will be created:
    
    ```
    export_dir_path:
      training_data:
        --> training_data:
            --> img
                --> <image name>.npy
            --> seg
                --> <segmentation name>.npy
            --> loc:
                --> <loc name>.nii.npz
      
      validation_data:
            --> img
                --> <image name>.npy
            --> seg
                --> <segmentation name>.npy
            --> loc:
                --> <loc name>.nii.npz
      test_data:
            --> img
                --> <image name>.npy
            --> seg
                --> <segmentation name>.npy
            --> loc:
                --> <loc name>.nii.npz
    ```

    Of course, there is not a single file in ct, seg, loc directories, but multiple files with corresponting names.
    </li>

    <li><b>normalization_function</b>, string, it is a single letter which represents normalization scheme. It can be: <b>"V", "F", "M"</b></li>    
    
    <li><b>number_of_samples_per_class</b>, int, this parameter defines how many samples of each class will be stored in loc files (.npz). In NNUnet they have set it to 10,000. I have put it to default value of None which means all FG locations for some class will be stored. This enhances the computing time, so feel free to set smaller number.</b></li>    
    </ul>

    <b>Output</b>:
    <ul>
    <li><b>organized directories and files</b> - as explained above.
    </ul>
   
    Script which implements it is named: ${\color{green}\text{"dataset\_formating.py"}}$, while the main implementation is actually in a class  ${\color{green}\text{"DatasetExport"}}$ which can be found in ${\color{green}\text{"DatasetPreprocessing/Utils"}}$ along with scripts  ${\color{green}\text{"normalization.py"}}$ and ${\color{green}\text{"cropping\_nnUnet.py"}}$.
  </details>

* <details> <hr> <summary> create_dataset </summary> 
    This script is main script that handels <b>dataset splitting</b>, <b>fingerprinting</b>, and <b>dataset export</b> all with one call.
    It requires four arguments to work:

    <b>Args:</b> 

    <ul>
    <li><b>input_dir_path</b> -- string, this is path to the input directory where the data is stored. It must have the following structure:
    
    ```
      input_dir_path:
      --> img:
          --> <image name#1>.nii.gz
          --> <image name#2>.nii.gz
          --> <image name#N>.nii.gz
      --> seg:
          --> <segmentation name#1>.nii.gz
          --> <segmentation name#2>.nii.gz
          --> <segmentation name#N>.nii.gz
    ```

    </li><hr>
    <li><b>export_path</b> --  string, Path where the data will be splitted. In that directory at the end of the script execution the user will see the following structure:
    
    ```
      export_path:
      --> <labels_file>.json
      --> training_fingerprint.json
      --> training_data:
          --> img:
              --> <training image name#1>.nii.gz
              --> <training image name#2>.nii.gz
              ...
              --> <training image name#N>.nii.gz
          --> seg:
              --> <training segmentation name#1>.nii.gz
              --> <training segmentation name#2>.nii.gz
              ...
              --> <training segmentation name#N>.nii.gz
      --> validation_data:
          --> img:
              --> <validation image name#1>.nii.gz
              --> <validation image name#2>.nii.gz
              ...
              --> <validation image name#N>.nii.gz
          --> seg:
              --> <validation segmentation name#1>.nii.gz
              --> <validation segmentation name#2>.nii.gz
              ...
              --> <validation segmentation name#N>.nii.gz
      --> test_data:
          --> img:
              --> <test image name#1>.nii.gz
              --> <test image name#2>.nii.gz
              ...
              --> <test image name#N>.nii.gz
          --> seg:
              --> <test segmentation name#1>.nii.gz
              --> <test segmentation name#2>.nii.gz
              ...
              --> <test segmentation name#N>.nii.gz
      --> formated_data:
          --> training_data:
              --> img:
                  --> <training image name#1>.npy
                  --> <training image name#2>.npy
                  ...
                  --> <training image name#N>.npy
              --> seg:
                  --> <training segmentation name#1>.npy
                  --> <training segmentation name#2>.npy
                  ...
                  --> <training segmentation name#N>.npy
              --> loc:
                  --> <training location name#1>.npz
                  --> <training location name#2>.npz
                  ...
                  --> <training location name#N>.npz
          --> validation_data:
              --> img:
                  --> <validation image name#1>.npy
                  --> <validation image name#2>.npy
                  ...
                  --> <validation image name#N>.npy
              --> seg:
                  --> <validation segmentation name#1>.npy
                  --> <validation segmentation name#2>.npy
                  ...
                  --> <validation segmentation name#N>.npy
              --> loc:
                  --> <validation location name#1>.npz
                  --> <validation location name#2>.npz
                  ...
                  --> <validation location name#N>.npz
          --> test_data:
              --> img:
                  --> <test image name#1>.npy
                  --> <test image name#2>.npy
                  ...
                  --> <test image name#N>.npy
              --> seg:
                  --> <test segmentation name#1>.npy
                  --> <test segmentation name#2>.npy
                  ...
                  --> <test segmentation name#N>.npy
              --> loc:
                  --> <test location name#1>.npz
                  --> <test location name#2>.npz
                  ...
                  --> <test location name#N>.npz
          --> normalization_fingerprint.json


    ```
    </li><hr>
    <li><b>labels_path</b> -- string, path to the labels json where all labels are defined. That json must be structured like this:
    
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

    </li><hr>
    <li><b>proc</b> -- integer, number of processes used to perfrom dataset export.</li><hr>
    </ul>

    <b>Output:</b> 
    <ul>
    </li><hr>
    <li><b>Organized directories and files</b> -- as explained above.</li><hr>
    </ul>


  <hr>
  </details>

Script which implements it is named: ${\color{green}\text{"create\_dataset.py"}}$, and it calls a bunch of other script from <i>dataset fingerprinting</i>, and <i>dataset preprocessing</i>.

<hr>

## DatasetExport class

Here I want to present logic behind <b>DatasetExport</b> class implemented in ${\color{green}\text{"DatasetPreprocessing/Utils/dataset\_export.py"}}$. The class handles all the exporting of the files, generating the directorires and subdirectories structure, performing cropping and normalization.

Cropping is done in  ${\color{green}\text{"DatasetPreprocessing/Utils/cropping\_nnUnet.py"}}$ which is an exact copy of the cropping script in the NNUnet, but with additional comments. The most important function in that script is: <b>crop_to_nonzero</b> which is the entry point for cropping.

Normalization is done in  ${\color{green}\text{"DatasetPreprocessing/Utils/normalization.py"}}$. I have implemented several normalization scheems:
    <ul>
    <li><b>NormalizeCTFromFingerprint</b> -- Z-normalization based on the provided path to fingerprint which is caluclated in <i>generate_dataset_fingerprint</i> script. This means that same mean and std is applied to all volumes. Also, the CT clipping is done to 0.05 and 0.95 percentiles. Code letter <b>F</b>.</li><hr>
    <li><b>NormalizeSegmentationsFromFingerprint</b> -- THIS IS NOT A NORMALIZATION, this only ensures that all labels presented
    in Segmentation map are the ones also presented in fingerprint file (generated by <i>generate_dataset_fingerprint</i> script). Why? Sometimes during the copying of the file, for some unknown reason to me a few pixels get wierd values such as 64352. So this normalization unsures that such labels are, by convention, put to 0(background). Code letter <b>F</b>.</li><hr>
    <li><b>NormalizeCTFromVolume</b> -- Z-normalization based on all voxels values presented in the given volume. Normalization is based on the per-volume-basis. Code letter <b>V</b>.</li><hr>
    <li><b>NormalizeCTFromMask</b> -- Z-normalization based on voxels values that are belonging to the foreground in the given volume. Forgeground voxels are voxels where segmentation mask is having label differnet from background and ignore label. Normalization is based on the per-volume-basis. Code letter <b>M</b>.</li><hr>
    </ul>


## Implementation

1. <b>DatasetExport</b> -- class which handels cropping, normalization and export of the dataset. We chose <b>"npz" and "npy"</b> format because it has <b>memmap</b> feature. This means that the whole volume (3D array) does not need to be loaded in the the memory, just part of it that is in our focus. This feature speeds up a lot loading and data processing. The methods in the class are as follows:

    <ul>
      <li><b>__init__</b> -- init method which collects data. It obtains segmentation maps and volumes in the provided directories. It also checks for validity of given normalization method.</li>

      <li><b>"__obtain_files"</b> -- Method which lists the files in path_to_files and extract all files which ends with desired file_extension. Basicaly list of files. Also it sorts them so it is always consistant (it ensures compatibility between segmentations and volumes.)</li>

      <li><b>"__obtain_labels_locations"</b> -- Method to obtain label location for each segmentation map and return dict where 
      key is label voxel value and value is list of locations. There is a parameter <b>number_of_samples</b> which limits how many random samples for each class will be stored. In NNUnet it is by defualt 10,000, so I have put the same. This is usefull when sampling classes later on in the dataset methods. Since there is usually a lot of background voxels, we do not obtain all of them, we obtain the average number of foregrond voxels per class as the number background locations. E.G. We have three classes 1: 100 voxels, 2:200 voxels, 3: 300 voxels, number of locations for backgorund is (100+200+300)/3 = 200. To obtain background voxels, I have implemented 2 methods based on the FG to whole image ratio. If ratio is smaller than 0.75, <i>__obtain_background_coordinates_small_ratio</i> is called, otherwise <i>__obtain_background_coordinates_big_ratio</i> is called.</li>

      <li><b>"__obtain_background_coordinates_big_ratio"</b> -- The ratio is big which means that we have a lot of FG voxels. So the loations for bacground are find in with simple mask where the input segmentation map is equal to the background label.</li>

      <li><b>"__obtain_background_coordinates_small_ratio"</b> -- Because the ratio is small, which means we have a lot of background, we sample locations on random and just check if the voxel value at selected loacation is equal to background label. If it is, save it, if not, randomly sample another location. The process repeates until targeted number of locations is obtained. This seems much faster thatn big ratio.</li>

      <li><b>"__obtain_surface_labels_locations"</b> -- this method has the same goal as <i>__obtain_labels_locations</i>, but it finds "blobs"-connected regions and extract only locations on border of the regions. My idea is to have surfaces separated for some cases where we are more interested in the surface regions to be segmented properly while we do not care that much for the middle of the segmentation volume. <b>Surfaces are still not used anywhere in the code. This is for some future projects.</b> </li>

      <li><b>"_export_case"</b> -- This methods is like a <b>core</b> of the DatasetExport class. It loads the images, cropt them by calling <b>crop_to_nonzero</b> function from <b>cropping_nnUnet.py</b>. Then it normalize the volume and segmentation mask by calling one of the set normalization methods. Just as a reminder, normalization functions are implemented in <b>normalization.py</b> script. It calls "__obtain" methods and, lastly, export everything in appropriate folders under distinctive names. Names are generated by splitting the input name with ".nii", as it is assumed that the volumes are stored in NIFTI. </li>

      <li><b>"export_data"</b> -- This methods is <b>entry point</b> of the DatasetExport. This methods creates all output dirs. It <b> DELETES</b> output directory and creates completely new tree of dirs. This methods collect the data and calls <i>_export_case</i> method and it also supports multiprocessing. 
      <li><b>"export_normalization_fingerprint</b> -- This method is optional, but highly recommended to call because it exports small config <span style="color:green;"> "normalization_fingerprint.json"</span> which stores all necessary attributes to recreate preprocesing. This is necessary for inference on the previously unseen data. 
      </li>
    </ul>

<hr>

2. <b>cropping_nnUnet.py</b> contains everything necessary for cropping the volume only to the usefull context. Since I work with CTs, the cropping is typically minor, but if we want to expand to MRI one day, this is necessary so I let it be. The following scripts are implemented:

    <ul>
    <li><b>"crop_to_nonzero"</b> -- <b>entry point</b> and only function being called when doing cropping. This function calculates non zero mask (all indices in <b>VOLUME</b> which are different from 0), then it makes minimum bounding box arround the nonzero mask and finally it crops both volume and segmentation mask to the bounding box.</li>
    
    <li><b>"create_nonzero_mask"</b> -- function which extract non-zero values from the given volume.</li>
    <li><b>"get_bbox_from_mask"</b> -- funciton which estimates minimum bounding box arround given mask.</li>
    <li><b>"bounding_box_to_slice"</b> -- changes bounding box to slice so it is easier to apply it to the volume. Just a simple transformation.</li>
    <li><b>"get_bbox_from_mask_np"</b> -- function which get bbox by using no.where which is slower than the implementation in <i>get_bobx_from_mask</i></li>  
    <li><b>"crop_to_bbox"</b> -- function which crops the volume to the bounding box.</li>
    </ul>

<hr>

3. <b>normalization.py</b> implements all normalizations mentioned before. It consists of four classes where each class is having init method which handels all necessary variables required by the normalization (in some cases it is empty). Then every class has a run method which applies the normalization to the given arguments. Arguments are typically volumes to which augmentation applies. Overview of the classes:

    <ul>
    <li><b>"NormalizeCTFromFingerprint"</b> -- init requires path to fingerprint. Code is <b>F</b>. Same std and mean to all volumes based on the average. It uses <b>"VOXEL_STATISTICS_SAMPLES"</b> by default.</li>
    <li><b>NormalizeSegmentationsFromFingerprint</b> -- <b>NOT A NORMALIZATION</b>, it simply check if all values in the segmentation mask are the real values of the labels provided by the user. This is just to avoid possible errors.</li>
    <li><b>NormalizeCTFromVolume</b> -- init does not require anything. It takes whole volume and normalize it based on its own std and mean. No "interaction" with other samples in the dataset. Code is <b>V</b>.</li>
    <li><b>NormalizeCTFromMask</b> -- init does not require anything. Normalize volume based on the foreground std and mean. No "interaction" with other samples in the dataset. Code is <b>M</b>.</li>
    </ul>
<hr>

### create_dataset script

This script removes all complexity and creates dataset from you with <b>minimum needed information</b>. That being said, it makes some assumptions which can only be changed manually in the script. So here are the assumptions

  <ul>
    <li><b>dataset splitting</b> -- dataset is split randomly with seed 1993 and in ration 80% of data to training set, 10% data to validation set and 10% data to test set.</li>
    <li><b>fingerprint</b> -- fingerprint is done only on training set!</li>
    <li><b>export and formating</b> --normalization method is set to <b>F</b> and number of samples exported for each class is <b>10000</b>.</li>
  </ul>
