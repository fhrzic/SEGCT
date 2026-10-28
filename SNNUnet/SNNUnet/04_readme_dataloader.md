# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Data/"}}$

As it can be seen, the Data is having three sub-directories and one file:

```
    Data:
        --> Dataloader:
            --> dataloader.py
        --> Sampler
            --> sampler.py
        --> Dataset
            -->NNUnet_Dataset
                --> dataset.py
            --> ModifiedNNUnet_Dataset
                --> dataset.py
            --> SingleSample_Dataset
                --> dataset.py
        --> verify_dataloader.py
        
```

Logic behind the scripts is following:

*  <b>Dataset</b> scripts are in charge of data loading. They implement a class which accepts path to data, load data, and on request <i>( invoke of __getitem__ method)</i> provides a patch and segmentation mask.

* <b>Sampler</b> scripts are in charge of sampling. It implements logic behind how are the classes sampled from the dataset. E.G. clases can be sampled uniformly or based on their occurance. It also samples foreground and background ratios per batch.

* <b>Dataloader</b> scripts wrap around everything and enables multiprocessing of the dataloading. This is a simple standard dataloader from Torch library adjusted for our need.

* <b>verify_dataloader.py</b> as the script name says, it is a helper script to see if dataloading is working fine. Good starting point to see/set the configs for Dataset,Sampler and Dataloader and set them up.

Sampler, Dataset and Dataloader come with its own config file which can be found in ${\color{green}\text{"Configurations/config\_data.py"}}$. Configs are basically classes which holds arguments important for Sampler, Dataset and Dataloader scripts. In the following text each of this scripts will be exaplained in details. 


Callable scripts are:

<ul>
<li><b>verify_dataloader</b> -- script for verifying dataset, dataloader and sampler. The idea is to manually change the script while the script provides a good example on how to implement the whole pipeline.</li>
</ul>

---

### Dataset script
<details>
Dataset scripts are the ones which are in charge of loading the data. They come with its own config file which can be found in <span style="color:green;">"Configurations/config_data.py"</span> class name: <span style="color:green;">"DatasetConfig"</span>.

There are three DatasetSchemes implemented:

1. In file <span style="color:green;"> "Dataset/NNUnet_Dataset/dataset.py"</span> is a class <span style="color:green;">PatchDataset</span>. The class is having following methods implemented:

    <ul>
    <li><b>"__init__"</b> -- initialisation method which as an argument accept instance of DatasetConfig class, which means all necessary parameters to load the data. It is in charge of several activities:
    <b>Locating files</b> (ct, properties and segmentation files), applying <b>white list</b> in case that we want to load only some file(e.g. We want to test some features of the dataset class), <b>limiting</b> number of samples in the dataset(in case that we want to train only on subset of samples in provided folders), <b>populating variables</b> used by other methods (e.g. label names), call <b> __populate_cases_and_classes</b> and  <b>__preload_volumes_and_segmentations</b> methods. 
    </li>

    <li><b>"__populate_cases_and_classes"</b> -- this method creates a dictionary which has as a key index of a case (cases from directory are put in a list), and as a value all labels presented in the observed case (obtained from properties file of a case). This method supports multiprocessing so it is calling method <b>_worker_for_obtaining_classes_in_case</b>.
    </li>
    <li><b>"_worker_for_obtaining_classes_in_case"</b> -- this is just a middleman for multiprocessing reasons. It <b>loads</b> loc file of the given case and return list of classes present in that case.</li>

    <li><b>__preload_volumes_and_segmentations</b> -- to speed up the process, especially in cases when the dataset is small, we can load CTs and SEG maps into the RAM (at least protion of them). So this method preloads in memory those cases which has the most classes in them. It loads them with <b>_worker_for_obtaining_volumes_and_segmentations</b> method because it supports multiprocessing. 
    </li>
    <li><b>_worker_for_obtaining_volumes_and_segmentations</b> -- method for loading CTs and SEG maps into the memory.</li>    
    <li><b>__getitem__</b> --  <span style="color:green;">"Main Method"</span> for loading the data. As an argument it gets dictionary which has two keys: index (which sample) and class (which class from that sample) it should sample-return. First, it checks if the sample is already <b>loaded</b> in memory with  <b>__preload_volumes_and_segmentations</b> method, if not it loads them with <b>__load_sample</b> method. After it, it obtains patch  by calling methods <b>__obtain_patch_coordinates</b> and <b>__get_patch</b>. Lastly, it applies augmentation provided and returns the patch and segmentation for selected patch to the caller of the method.
    </li>    

    <li><b>"__load_sample"</b> -- method which load sample with memorymap mode on. This method could be potentially merged with <b>_worker_for_obtaining_volumes_and_segmentations</b> because it do same thing, but I keep it separated for cleanliness</li>

    <li><b>"__obtain_patch_coordinates"</b> -- this method obtain patch coordinates and performs padding. This is a bit convoluted so please see the code for detail explanation how it is done. The important note is: it gets volume, and returns top left and bottom right corner of the patch. The patch center is selected based on the location of the class from the loc file of the observed sample. In case that random patch is needed, than the class is not selected, it is just a random voxel inside the volume (it can be foreground/class or background)</li>

    <li><b>"__get_patch"</b> -- method "crops" out the patch from the coordinates obtained by <i>"__obtain_patch_coordinates"</i> script. It crops simultanously volume and segmentation map. It ensures the desired patch size by additional padding.</li>

    <li><b>"__len__"</b> -- method returns number of samples in datasets. It is required by some samplers.</li>

    <li><b>"get_cases_and_classes"</b> -- method returns dictionary which is generated by <b>"__populate_cases_and_classes"</b> method. It is requirement for some samplers. </li>
    </ul>
2. In file <span style="color:green;"> "ModifiedNNUnet_Dataset/dataset.py"</span> is a class <span style="color:green;">PatchDataset</span>. The class is having following methods implemented: TODO

3. In file <span style="color:green;"> "SingleSample_Dataset/dataset.py"</span> is a class <span style="color:green;">SingleSampleDataset</span>. The class is having following methods implemented:
    <ul>
    <li><b>"__init__"</b> -- initialisation method which accepts instance of <b>DatasetConfig</b> class which hold all necessary data to build the dataset. This dataset works on principle that it takes sample by sample from the given directory and then create batches based on non-overlapping patches of desired shape. The last patch if it does not fit is pushed back into the volume. This is a perfect dataset for <b>validation</b> and it takes into a count all patches inluding those with only background. Besides obtaining list of files, it also calls methods <b>__obtain_background_and_ignore_labels</b>(to retreive background and ignore label values from the fingerprint) and <b>__preload_volumes_and_segmentations</b> (in case that flag to keep in memory is set to a value).</li>
    <li><b>"__obtain_background_and_ignore_labels"</b> -- method which based on provided path to the fingerprint (read out from the config file) obtains <b>background</b> and <b>ignore_label</b> values necessary for further script work.</li>
    <li><b>"__preload_volumes_and_segmentations"</b> -- in case that the flag <b>keep_in_memory</b> is set to a value, that amount of cases will be loaded into RAM. This is can be done in multiprocessing manner which rised a need to implement <b> _worker_for_obtaining_volumes_and_segmentations</b> method. Nothing fancy, just loading and keeping in RAM.</li>
    <li><b>" _worker_for_obtaining_volumes_and_segmentations"</b> -- script which loads ct and seg map from the given index (path).</li>
    <li><b>"__symetrical_padding"</b> -- Method which adds padding symetrically (symetrical as possible) before and after the volume 
        until it is fully devisible the desired_shape is reached. If the devisable\_by\_patch is True, then it is padded more until it is devisable by patch. It returns padded tensor as well as the paddings in every direction.
    </li>
    <li><b>"__split_volume_into_patches_no_padding"</b> --  Split the 3D volume into (non-)overlapping patches of shapeprovided by confing class to <b>__init__</b> method. But instead  of zero padding the volume to the closes devisable shape, it does reverse. It creates all normal patches, then it makes patches on the residuals. E.G. On 1D example let's say that volume is 130px long and patch is 100. It will make two patches frist one from 0-100, and second from 30-130. It Will not zero pad the volume to 200 and then make two patches: from 0-100 and 100-200.</li>

    <li><b>" __getitem__"</b> -- main method to iterate over the samples. First it check if the sample is preloaded, if it is not, then it loads sample by calling <b>load_sample</b> method. After it it padds the method with symethrical padding by invoking <b>__symetrical_padding</b> method, after which the padded volumes are split into the patches by <b>__split_volume_into_patches_no_padding</b> method. Padding is necessary for the volumes which are smaller than the desired patch size.</li>
    <li><b>"__len__"</b> -- returns number of samples in the dataset (number of cases in the provided directory).</li>
    </ul>
</details>

---

### Sampler script
<details>
Sampler scripts are the ones which are in charge of smapling the data from dataset. They come with its own config file which can be found in <span style="color:green;">"Configurations/config_data.py"</span> class name: <span style="color:green;">"SamplerConfig"</span>.

There are three samplers implemented:

1. In file <span style="color:green;"> "Data/Sampler/sampler\_nnUnet\_style.py"</span> is a class <span style="color:green;">NNUnetSampler</span>. This sampler is connected to <span style="color:red;"> "Dataset/NNUnet_Dataset/dataset.py"</span>. The class is having following methods implemented:

    <ul>
    <li><b>"__init__"</b> -- initialization method which as an argument accept instance of the SamplerConfig class, which means all necessary parameters to build the sampler. Important parameters are <b>batch_size</b>, <b>fg_sample_ratio</b>, <b>dataset</b>. Each of this variables control some aspect of sampling. <b>batch_size</b> defines number of samples yielded in each iteration, <b>fg_sample_ratio</b> controls how many samples in the batch are required to have patch sampled around the class(100% chance that patch contain some voxels of a foreground), minimum for fg_sample_ratio is 1, so if the batch size is set to 2, and fg_sample_ratio is 33% (0.33), it will be the same as the ratio is 0.5. <b>dataset</b> is a instance of class implemented in  <span style="color:green;"> "Dataset/"</span> dir files (please see details of dataset script). Also, in init we obtain a dictionary via dataset method <b>get_cases_and_classes</b> which contain classes presented in every case.

    <li><b>"__iter__"</b> -- custom logic which returns all necessary parameters to sample from dataset. In this sampler, it returns <b>index</b> of a sample from the dataset and <b>class</b> for that sample where the patch will be centered. So if the batch size is 8, and fg_sample_ratio is set to 0.25, that means that method will select 8 random cases (indexes), and then for 2 of the selected cases will have uniformly selected class from the classes available in that case. Other 6 cases will as <b>class</b> have value of -1 which indicates that random position will be selected. 
    </li>

    <li><b>"__len__"</b> -- this method returns the number of batches in the dataset which is infinity</li>
    </ul>

2. In file <span style="color:green;"> "Data/Sampler/sampler\_nnUnet\_modified.py"</span> is a class <span style="color:green;">CustomBatchSampler</span>. This sampler is connected to <span style="color:red;"> "Dataset/ModifiedNNUnet_Dataset/dataset.py"</span>.The class is having following methods implemented:

    <ul>
    <li><b>"__init__"</b> -- initialization method which as an argument accept instance of the SamplerConfig class, which means all necessary parameters to build the sampler. Important parameters are <b>batch_size</b>, <b>fg_sample_ratio</b>, <b>dataset</b>, <b>class_sampling_dict</b>, and <b>background_as_random</b>. Each of this variables control some aspect of sampling. <b>batch_size</b> defines number of samples yielded in each iteration, <b>fg_sample_ratio</b> controls how many samples in the batch are required to have patch sampled around the class(100% chance that patch contain some voxels of a foreground), minimum for fg_sample_ratio is 1, so if the batch size is set to 2, and fg_sample_ratio is 33% (0.33), it will be the same as the ratio is 0.5. <b>dataset</b> is a instance of class implemented in  <span style="color:green;"> "Dataset/"</span> dir files (please see details of dataset script). Also, in init we obtain a dictionary via dataset method <b>get_cases_and_classes</b> which contain classes presented in every case. The <b>class_sampling_dict</b> controls the ratio at which the classes will be sampled. E.G. If we provide <i>None</i> as the value, the unifrom sampling will be done which means that class will be uniformly selected, however if dict is provided, then that dict must have class:probability pairs at which the classes are sampled. It is also possible to provide a path to fingerprint where the probabilities are claculated based on their foreground ratios to the volume(the bigger the class is, the higher probability to get sampled it has). Last parameter: <b>background_as_random</b> controls how the background sample will be done (in the given example, 6 cases which are randomly selected). If the background_as_random is set to a value e.g. 0.4, then 0.4 of 6 cases (2) will be sampled with true random location, when 4 cases will be sampled from the backgroun locations of given case ensuring that the sampled patch contains background. 

    <li><b>"__obtain_distribution_dict_from_fingerprint"</b> -- method which if the path to the fingerprint is provided will calculate probabilities for each class to be sampled based on the averaged normalized ratio of that class in the volume.</li>

    <li><b>"__iter__"</b> -- custom logic which returns all necessary parameters to sample from dataset. In this sampler, for each sample it returens two parameters: <b>foreground</b> which can be True or False, and <b>class</b> which can be -1, -2 or a valid class index. Foreground defines based on the <b>fg_sample_ratio</b> for each sample in the <b>batch_size</b>. Lets say that batch size is 6 with fg_sample_ratio set to 50% (0.5), this means that 3 cases will have foreground set to True and 3 cases foreground set to False. Now, if foreground is set to True, it means that we need to sample a class for that case. In this sampler, the class is selected based on the <i>sampling_distribution</i> parameter (unifrom is default, but it can be redefined with __obtain_distribution_dict_from_fingerprint, or by provided dict with sampling probabilities) and stored in <b>class</b> parameter. Dataset when recieveing this two params for each sample, samples random case which contain provided class in case that foreground is set to true. This way we ensure that (i) all classes are equaily seen by model, and (ii) all cases containing the requested class are uniformly sampled (and hence seen by the model). In case that foregorund parameter is set to False, then the class can be either <b>-1</b> or <b>-2</b>. If it is -1, random case with random location is sampled. If it is -2, random case with background as a label will be selected, and then from that case, random location with background will be sampled as a center of a patch. At the end, sampler yields a list of batch_size paramteres (foreground and class) based on which dataset samples the patches.
    </li> 

    <li><b>"__len__"</b> -- this method returns the number of batches in the dataset which is infinity</li>
    </ul>

3. In file <span style="color:green;"> "Data/Sampler/sampler\_for\_validation.py"</span> is a class <span style="color:green;">ValidationSampler</span>. This sampler is connected to <span style="color:red;"> "Dataset/ModifiedNNUnet_Dataset/dataset.py"</span>.The class is having following methods implemented:

    <ul>
    <li><b>"__init__"</b> -- initialization method which as an argument accept instance of the SamplerConfig class, which means all necessary parameters to build the sampler. Important parameters is <b>number_of_samples</b> and <b>dataset</b>. The idea is to obtain from the dataset with its <b>__len__</b> method number of cases, and than devide number_of_samples evenly through cases so that from each case we samples equal number of foreground cases. E.G. If there are 5 cases and number of samples is 100, from each case randomly 20 patches will be sampled which will be provided for validation. The idea behind this sampler is to sample only foreground samples on random uniformly from the cases provided in the validation subfolder. It also obtain classes for each case so it knows which classes can be sampled from which case.

    <li><b>"__iter__"</b> -- it samples for every case equal number of samples. It also unfiromly choses from classes available in the case so it is balancing over underrepresented classes. At the end, it provides for one case at the time a big batch of all patches sampeled for that case.
    </li>

    <li><b>"__len__"</b> -- this method returns the number of cases in the validation dataset</li>
    </ul>
</details>

---

### Dataloader script
<details>
Dataloaders scripts are the ones in charge of making loading multiprocessed and connecting dataset with sampler. Scripts are implemented in <span style="color:green;">Dataloader/dataloader.py</span>. They come with its own config file which can be found in <span style="color:green;">"Configurations/config_data.py"</span> class name: <span style="color:green;">"LoaderConfig"</span>. Dataloader calls dataloader from <a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader" title="Torch Dataloader">Torch Dataloader</a> where the parametrs provided in <b>LoaderConfig</b> are basically required by the torch dataloader.

Each dataloader have the following parameters which needs to be defined:

<ul>
<li><b>"dataset"</b> -- dataset to be loaded</li>
<li><b>"sampler"</b> -- defined sampler (in our case it is batch_sampler)</li>
<li><b>"number_of_workers"</b> -- number of subproceses for data loading</li>
<li><b>"prefetch_factor"</b> -- number of batches loaded in advance by each worker</li>
<li><b>"use_gpu"</b> -- if True, the data loader will copy Tensors into device</li>
<li><b>"device"</b> -- the device to pin_memory to if pin_memory is True. In this case pin_memory == use_gpu</li>
<li><b>"collate_fn"</b> -- merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset. There is a custom function implemented which handles the implemented datasets.</li>
</ul>


There are two loaders implemented: 

1.  <b>infinite_dataloader</b> -- as the name says, it is an infitinite dataloader, it loads and loads and loads... It needs to be 
controled by and external loop. This dataloader requiers  <span style="color:green;">custom_collate_fn_for_infinite_dataloader</span> function implemented in the same py file where the dataoloader can be found: <span style="color:green;">Dataloader/dataloader.py</span>. Namely, if dataset returns list and not only tensors, which is the case if we choose to use deep-supervision augmentation which returns list of tensors on different scales, we need to create custom collet function which will stack the list in tensors and return them in a format desired by the torch. In simple terms: stack the tensors of same shape so that torch can do its "magic".

2. <b>one_sample_dataloader</b> -- Just for the consistency reasons. This  dataloader returns one sample at the time which is convinient for validation and test sets. It uses predifined <b>batch_size</b> of 1.

</details>

---

### verify_dataloader.py script

The <b>verify_dataloader.py</b> script serves the purpose of validating that everything is connected right and that our dataloading is working correctly. It is mainly designed for testing purposes and as such is <b>NOT CALLED</b> anywhere in the program nor is a requierement for pipeline to work. In order to call this script, a command is added <span style="color:green;">verify_dataloader</span>. However, the command is not requireing any arguments and if someone wants to tune it, it must fix code inside the script. In this sample method we build everything for dataloader with examples for each of three components: Sampler, Dataloader and Dataset. Also, it exports the configs and have function <i>build_data_loader_from_config</i> which demonstrates how to rebuild the whole dataloading pipeline from exported config files. 

The problem to overcome when rebuiliding from the configs is that e.g. dataloader has two propertis which are actually classes: dataset and sampler. So in confing files the <b>names of the classes</b> are saved so we know what class was given as propery to the dataloader. But we still need to build the appropriate class based on that class config. So my idea se when rebuilding from confing files to first build Augmentation(no classes as propertis), Dataset (augmentation is class), Sampler(dataset is class), Dataloader (dataset, sampler, and collate function are classes). I know it is wierd, but in the NNUnet the authors had a function which seeks for the class name everywhere in the code which is rather confusing!

<b>Args</b>:
  <ul>
  <li><b>export</b>, string, path to the dir where the exports will be make. It is safe to delete this file after.</li>
  <li><b>device</b>, string, <i>cuda:0</i>, <i>cuda:1</i>, <i>cpu</i>... Which device to use to load data. Since the GPUs can be occupied,
  it is decided to put this as a separate argument to prevent unintional loading of data on GPUs. Default is <i>cpu</i>.</li>
  </ul>

  <b>Output</b>:
  <ul>
  <li><b>config files</b> - saved into different <i>".json"</i> files.
  </ul>


---

## CONFIG FILES

In <span style="color:green;">Configurations/config_data.py</span> are confinguration classes for all above mentioned components. Each confing class typically contains three methods: 

<ul>
<li><b>"__str__"</b> -- to print confing. Classes are printed as names.</li>
<li><b>"export_to_json"</b> -- method which accepts <b>export_path</b> as an argument which is a path to the json file where the confing will be saved. DISCLAIMER: one more time, propertis which are either class or function are stored as the the names of the classes and functions.</li>
<li><b>"import_from_json"</b>-- method which accepts <b>import_path</b> as an argument which is a path to the json file where the confing is stored. DISCLAIMER: one more time, propertis which are either class or function are stored as the the names of the classes and functions.</li>
</ul>

Each class confing class also have a set of propertis which can be set. The properties are as follows:

---

<b>Class:</b> <span style="color:green;">DatasetConfig</span>
<details>
<ul>
<li><b>"ct_path"</b> -- DEFUALT: None -- path to the ct directory contatining nifti files.</li>
<li><b>"seg_path"</b> -- DEFUALT: None -- path to the segmentation maps(niftis) directory.</li>
<li><b>"loc_path"</b> -- DEFAULT: None -- path to the potential locations.</li>
<li><b>"data_fingerprint_path"</b> -- DEFAULT: None -- path to dataingerprint file generated by generate_dataset_fingerprint script.</li>
<li><b>"patch_shape"</b> -- DEFAULT: (128, 128, 128) -- size of a patch in 3 dimensions (x,y,z). This is the input size that is sampled from the volume. Some augmentations reduces the input size additionaly (center crop), and some requires larger input size to prevent running out of volume space (rotation). So, this patch_shape is <b>not the output patch shape</b>, it is the shape that is sampled from the volume before augmentation. </li>
<li><b>"minimum_fg"</b> -- DEFAULT: 0.01 -- For some datasets it defines the miniumm percentage of the voxels which needs to be present in the sampled patch that belongs to the foreground when sampling for foreground. <span style="color:red;">Some dataset only.</span></li>
<li><b>"multiprocessing"</b> -- DEFAULT: None -- Number of processes used when preloading or acquiring locations. If set to None, no multiprocessing is used.</li>
<li><b>"augmentation_method"</b> -- DEFAULT: None -- instance of the class which performes augmentation.</li>
<li><b>"sample_limit"</b> -- DEFAULT: None -- in case that one wants to limit the number of samples, this parameter cut the number of samples to the set value. Usefull when testing the sampling (less cases results in less variety which makes inspection much easier).</li>
<li><b>"force_valid_locations"</b> -- DEFAULT: False -- if set True, only patches which are fully inside the volume will be 
    obtained. Also, if the patch is outside the volume, it would be "pushed" inside. <span style="color:red;">Some dataset only.</span></li>
<li><b>"include_background_as_class"</b> -- DEFAULT: False -- if set to True, background will be treated as a class which means that patches with the center set to a voxel which belongs to background would be sampled same as any other class.</li>

</ul>
</details>

---

<b>Class:</b> <span style="color:green;">SamplerConfig</span>
<details>
<ul>
<li><b>"batch_size"</b> -- DEFUALT: 8 -- number of samples in batch.</li>
<li><b>"fg_sample_ratio"</b> -- DEFUALT: 0.33 -- number of samples in batch (in percentage) which will be samples as a foreground (center voxel of the patch will belong to a class).</li>
<li><b>"number_of_samples"</b> -- DEFAULT: None -- in case of validation sampler, this is used to set number of samples (patches) to be sampled.</li>
<li><b>"dataset"</b> -- DEFAULT: None -- instance of a class which implements dataloading.</li>
<li><b>"class_sampling_dict"</b> -- DEFAULT: None -- dictionary where key is class label and value is probability to sample this class. If it is string (it must be path to fingerprint) it will caluculate the probaiblity to sample each class based on the class presence in volumes (the higher median value of voxels of a class is in the volume, the higher chance to sample that class). <span style="color:red;">Some samplers only.</span></li>
<li><b>"background_as_random"</b> -- DEFAULT: None -- when the non-foreground sample is sampled, the patch center is selected as random voxel. With this parameter it can be forced to be a background (paremeter is set to the probability that the background voxel is demanded. So if it ise set to 0.25, it menas that out of the samples which are sampled as non-foreground, 25% will have backgournd voxel as the center voxel of a sampled patch). In summary, this paramter forces to sample background in some cases if we want to do it.<span style="color:red;">Some samplers only.</span></li>

</ul>
</details>

---
<b>Class:</b> <span style="color:green;">LoaderConfig</span>
<details>
<ul>
<li><b>"dataset"</b> -- DEFUALT: None -- instance of a class which implements dataloading.</li>
<li><b>"sampler"</b> -- DEFUALT: None -- instance of a class which implements sampling.</li>
<li><b>"number_of_workers"</b> -- DEFAULT: 0 -- number of subproceses used for data loading.</li>
<li><b>"prefetch_factor"</b> -- DEFAULT: 2 -- number of batches that each worker prefatches</li>
<li><b>"use_gpu"</b> -- DEFAULT: False -- if set to True, the data will be transfered to the selected device (device argument).</li>
<li><b>"device"</b> -- DEFAULT: "cpu" -- device to which the data will be loaded</span></li>
<li><b>"collate_fn"</b> -- DEFAULT: None -- custom collate function to build batches. It is necessary if all returned data can not be concatinated which is the case if we use deep supervision where each sample also returns segmentation maps on different scales.</li>
</ul>
</details>

---