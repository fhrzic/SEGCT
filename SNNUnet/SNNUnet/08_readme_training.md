# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Training"}}$ directory.

As it can be seen, the directory <span style="color:green;"> "Training"</span> the following scripts can be found:

```
    Training:
        --> training_class.py
        --> manual_training.py
        --> manual_training_with_arguments.py
        --> configuration_based_training.py
```

The scripts contents/purposes are as follows:

<ul>
<li>"training_class" -- is a main script for training. It contains and implements <span style="color:green;">MainTrainingClass</span> which is literally the class that perform training. It is like a glue between: training and validation dataloaders, model, optimezer and losses.</li>
<li>"manual_training" -- script which shows how to perform training. It includes everything: data confings, model building, setting optimizers, training configs... It is nice script that demonstrates how everything is put together. Name <b>"manual"</b> is put here because, although it can be run <b>"as-it-is"</b> with 0 parameters change, it can be inspected and changed manually and get a feelling how everything is put together. It is quite lenghty, but from my point of view - really intuitive.</li>
</ul>

---

## training_class.py 
<details>
Script <span style="color:green;"> "training_class.py"</span> located inside directory <span style="color:green;">Training</span> implements <span style="color:green;">MainTrainingClass</span>. Here is the list of all methods which are implemented in the class:

<ul>
<li>"__init__" -- method which is in charge of initializing: <b>model</b>, <b>loss</b>, and <b>optimizer</b>. Each of the mentioned components is initialized within its own method: <b>"__obtain_model"</b>, <b>"__obtain_loss"</b>, and <b>"__obtain_optimizer</b> respectively. Besides initializing the mentioned components, this method instances gradiane scaler, create output directory and set dataloaders available through the class. Inputs to this method are as follows:
<ul>
<li>train_dataloader -- instance of the dataloader which loads training data. Typically this is going to be <b>infinite_dataloader</b> implemented in <b>Data/Dataloader/dataloader.py</b> script. To obtain dataloader, user must specify configs files for dataset, sampler and dataloader. For more info on how to build dataloader please check <b>manual_training.py</b> pr <b></b></li>
<li>valid_dataloader -- instance of the dataloader which loads validation data. Typically it is <b>one_sample_dataloader</b> or some similar variation to it. It is important that it is not the infinite dataloader!</li>
<li>model_config --  instance of config class which hold all necessary data to implement model. The config class can be found in <b>Configurations/config_model.py</b>. </li>
<li>training_config -- is an instance of config class which hold all necessary propoertis which defines training. The training confing is implemented in  <b>Configurations/config_model.py</b> and more words about it will be given in the following sections.</li>
</ul>
</li>
<li>"__obtain_model" -- method which obtain model(return torch.nn.Module). Currently tested model is <b>SNNUnet</b> while <span style="color:red;"><b>VNet</b> is implemented but <b>not tested!</b>.</span> The method simply returns the model build from the provided config and it also checks via assert if the model is implemented(method knows how to implement the model). Therefore, if someone wants to add a new model, this method must be updated accordingly. Beside the argument <b>model_config</b> which hold all the necessary arguments to build the model, the method also accepts string which define <b>device</b> on which the model will be loaded. E.G. "cuda:0", "cuda:1", "cpu"... </li>
<li>"__obtain_loss" -- method which obtain loss based on the loss config. It returns both training and validation losses based on the loss function configs which is provided in <b>training_config</b>. There are four losses implemented in total, well three because deep supervision loss is not a new loss but combination of two:
<ul>
<li>"nll" -- <b>simple config</b> because it requires only the name of the loss and it is implemented in <a href=https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html>torch.</a>This config can be provided by <span style="color:green;">LossConfigFromTorch</span> which can be found in <b>Configurations/config_loss.py</b>.</li>
<li>"cross_entropy" -- <b>simple config</b> because it requires only the name of the loss and it is implemented in <a href=https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>torch.</a> This config can be provided by <span style="color:green;">LossConfigFromTorch</span> which can be found in <b>Configurations/config_loss.py</b>.</li>
<li>"DCandCE" -- <b>complex config</b> which can be found in the <b>Configurations/config_loss.py</b> in the class <span style="color:green;">LossConfig_DCandCE</span>. It is called complex because it requires additional parameters - it is not just a torch implementation.</li>
<li>"dsLoss" -- <b>complex config</b> which can be found in the <b>Configurations/config_loss.py</b> in the class <span style="color:green;">LossConfigDeepSupervision</span>. It is called complex because it requires additional parameters - it is not just a torch implementation. Also, this can not be considered as a pure loss because it is more like a "wrapper" arround other loss function and applies it just on multiple auxiliary outputs. </li>
</ul>
</li>

<li>"__obtain_optimizer" -- method which obtain optimizer based on the <b>optimizer_params</b> in the <b>Training_config</b>. <b>optimizer_params</b> are actually givven in script <b>Configurations/config_optimizer.py</b> and hold all necessary information about optimizer and scheduler. As the optimizer, currently only one optimizer is implemented under the name <span style="color:green;">SGD</span> which stands for StohasticGradientDescent. The default values are set to be the same as in NNUnet. This method also returns scheduler. The implemented schedulers from the <b>Scheduler/</b> directory are <span style="color:green;">"fibo"</span> and <span style="color:green;">"poly"</span>. It also suporte <b>None</b> if the scehduler someone does not want to use scheduler. </li>
<li>"start_training" -- <b>ENTRY POINT</b> of model training. After the class is initialised, this method should be called to start the training. Here are the steps that this script is performing:
<ol type="i">
<li>Although the Accelerator for faster training is implemented, <span style="color:red;">It has never been tested.</span>. But here it is initalised!</li>
<li>If the <b>checkpoint</b> is provided in model training, then the model is loaded and the epoch is set to be the one saved in model.</li>
<li>The training starts: for each epoch the method <b>train_one_epoch</b> is called.</li>
<li>After the model has been trained for one epoch, <b>validate_one_epoch</b> is called</li>
<li>Results are printed to the user for examination.</li>
<li>Based on the saving interval, the model is saved. Model is always saved if it achieves better score on the validation set!</li>
<li>Final training log is exported in file named <b>training_log.json</b> which hold all losses and validation metrics during the training. The file is saved in the <b>Checkpoints directory</b>.</li>
</ol>
</li>
<li>"train_one_epoch" -- as the name of method says, it trains model for one epoch. Since it is usually trained by dataloader which incorporates the infinite sampler. Infinite sampler does not have a limit on number of batches per epoch, hence the number of how many batches during the one epoch model will see is given by <b>batch_per_epoch</b> parameter in <b>training_config</b> which is provided to the <b>__init__</b> method. At the beggining of this method, storage is initialized where the result of every batch will be sotred for monitoring. This serves the purpose for logging and reporting. Then the method goes into <b>for loop</b>(it could be while...) which calls dataloader to provide in order to obtain batch. After the batch is obtained, it tries to use accelrator <span style="color:red;"> which is not tested to work - it should </span>. It transfers everything to <b>device</b> provided in <b>training_config</b>. Then it obtains prediction with respect to the <b>mixed_precision</b> parameter, after which the loss is calculated. When calculating loss the <b>training_one_epoch</b> takes care of appropriate input formats based on the selected loss function. Once when the loss is calculated, it is backpropagated through the model applying gradient scaler. Gradient scaler is created in <b>__init__</b> method, which is different from <b>accelearator</b> and <b>mixed_precision</b> flags which are provided in <b>training_config</b>. After the loss has been propagated, the loss storage is updated with the loss of the provided batch. At the end, this method <b>returns</b> the loss storage which is holding loss for every batch to its caller. </li>
<li>"validate_one_epoch" -- this method validates one epoch, whatever that epoch is. I am saying that because there are two implemented ways how the validation epoch is created. Both have one thing in common: they scane the provided valdiation directory and inspect <b>one case at the time. The first implemented is </b><span style="color:green;">SingleSampleDataset</span> which devides the input volume into non-overalaping patches and calculate metrices for each patch. This is realistic measurment of the model prediction but it takes into account background which in cases where there is a lot of background might not be a good idea. Naturally, all reported metrices are averaged across all patches and cases. As the oposition "single sample approach", the second one is based on </b><span style="color:green;">PatchDataset</span> but instead of using <b>NNUnetSampler</b> it utilizes <b>ValidationSampler</b> which selects only random foregrounds from the provided validation cases (provided as directory). Nontheless, the method returns dictionary with <b>mean recall, mean precision, mean f1, and mean loss</b> which is than in <b>start_training</b> printed to user and added to the storages for logging.</li>
<li>"__save_model" -- method in charge of saving the model. It saves always the </b><span style="color:green;">last checkpoint under tha name "last_checkpoint.pth"</span> and if necessary best model under the name </b><span style="color:green;">best_model_checkpoint.pth</span>. Simply, everything needed to reconstruct and restart the training <b>(model's weights, optimizer, scheduler, epoch, and gradient_sclaer)</b>.</li>
<li>"__load_model" -- method which loads <b>model's weights, optimizer, scheduler, and gradient_sclaer</b> states from the provided checkpoint path. The method also returns the <b>epoch</b> (which is also saved in the state dict of the chackpoint).</li>

</ul>
</details>
---

### config_training.py script
<details>

<span style="color:green;"> "config_training.py"</span> located in <span style="color:green;"> "Configurations"</span> directory contains all the necessary attributes/properties to train the NNUnet. It is implemented in the <span style="color:green;"> "TrainingConfig"</span> class.  The attributes in the model are as follows:


<b>Attributes:</b>
<ul>
    <li><b>"epochs"</b> -- integer -- integer specifying number of epochs.</li>
    <li><b>"batch_per_epoch"</b> -- integer -- integer specifying number of batches per epoch.</li>
    <li><b>"batch_size"</b> -- integer -- integer specifying batch size.</li>
    <li><b>"optimizer_params"</b> -- class -- instance of OptimizerConfig which is specifying optimizer and scheduler.</li>
    <li><b>"accelerator"</b> -- bool -- in case that we want to use acclerate libray (paralle computing). <span style="color:red;"> Although it is implemented it was not tested properly. </span></li>
    <li><b>"save_model_interval"</b> -- integer -- integer specifying after how many epoch the model will be saved.</li>
    <li><b>"training_loss_function"</b> -- class -- instance of the loss which will be used for model training.</li>   
    <li><b>"validation_loss_function"</b> -- class -- instance of the loss which will be used for model validation.</li>
    <li><b>"device"</b> -- string -- device on which the model will be trained.</li>
    <li><b>"output_dir"</b> -- string -- path to the directory where the output will be stored. <span style="color:green;"> It overrides directory!</span></li>
    <li><b>"resume_from_checkpoint"</b> -- string -- path to the lastk <b>"*checkpoint.pth"</b> file produced by <b>training_class</b> (method "__save_model")</li>
    <li><b>"gradient_scaler"</b> -- bool -- if set to true, during the backpropagation the gradient scaling will be used preventing too big gradients. </li>
    <li><b>"mixed_precision"</b> -- bool -- if set to true, model will be sed to mixed precision which means 16bit weights instead of 32 bit weights.</li>
</ul>

<br>

Config also supports saving and loading, as well as printing the attributes through methods:

<ul>
<li>"export_to_json" -- method which exports all attributes to the json file. <span style="color:red;"> For classes only their names will be stored -- class instance is imposible to be stored in json!</span></li>
<li>"import_from_json" -- method which imports all the attributes from the json file to the class. Remember, for classes such as <b>training_loss_function</b> only the class name is stored. One still need to provide a class, but at least he knows which class was implemented. In the configs that are outputed during the training, the correct loss function confing file should be present.</li>
<li>"__str__" -- prints all attribtus of the <b>TrainingConfig</b> class with classes supstitued by the class names.</li>
</ul>

</details>

---

### manual_training.py script

<details>
The script <span style="color:green;">manual_training.py</span> is located at <span style="color:green;">Training</span> directory and it implements minimum working example of the produced pipline. It is used for testing the training. It is called by the <span style="color:green;"><b>run_preset_pipeline</b></span> which exports the dataset and run the training on it. So this script can be executed manually and serves the purpose to test if everything is "glued" properlly together. It demonstrates how to correctly specify all confings for loss functions, models, dataloaders and how to run the traning. To be convenient, it accepts four arguments and can be run from the comand line:

<ul>
<li>"export_dir_path" -- string invoked by <i>--export</i> flag which specifies directory where the outputs of the training will be stored. There are two output directories created withing the specified output dir(which is <b>>overriden!</b>): <b>"Configs"</b> and <b>"Checkpoints"</b>. In the Configs directory all configurations necessary for training are stored while in Checkpoints dir all the models checkpoints are being saved.  </li>
<li>"device" -- string invoked by <i>--device</i> flag which tells on which device the model will be trained. E.G. "cpu", "cuda:0", or "cuda:1"</li>
<li>"path_to_data_fingerprint" -- string invoked by <i>--fingerprint</i> flag which is the path to the data fingerprint. This is necessary in order to build model. </li>
<li>"data_path" -- string invoked by <i>--data</i> flag which is a path to the directory where the <span style="color:green;">formated data</span> is exported. This means that the targeted directory expected structure is as follows:

```
    formated_data:
        --> training_data
            --> seg
            --> loc
            --> img
        --> validation_data
            --> seg
            --> loc
            --> img
        --> test_data
            --> seg
            --> loc
            --> img
```
This structure is obtained by the scripts in <b>DatasetPreprocessing</b>
</li>
</ul>

All in all, the script trains the model from the provided data with minumum context, so it can be considered as <b>ENTRY POINT</b> for model's training! If someone wants to test different setups, this is quite usefull script to have. <span style="color:red;">Just be carefull: changes in this script also impact other commands/scripts which calls the main of this script!</span>. One such script is <b>run_preset_pipeline</b>  which can be found in <span style="color:green;">RunScripts</span> directory.
</details>

---

### manual_training_with_arguments.py script
<details>
This script is <b>REPLICA</b> of <span style="color:green;">manual_training.py</span> and is also located in <span style="color:green;">Training</span>. What is the purpose of haveing identical script? I want to keep manual_training.py clear, and if someone wants to control any aspect of training via external arguments, well manual_training_with_arguments.py is a script to do so. For instance I have added 2-3 arguments to it just to show how it can be done (E.G. epochs argument controling number of epoch for model to be trained).
</details>

---

### configuration_based_training.py script
<details>
This script serves in scenario when we conducted one training, and then want to conduct another training but with slightly different parameters. So instead of changining everything in the script, one can modify files in <b>Configs</b> of the conducted research and run script <span style="color:green;"><b>run_training_from_config</b></span> which will be really helpfull. <span style="color:red;">It works, but it needs to be tested in details!</span.>
</details>