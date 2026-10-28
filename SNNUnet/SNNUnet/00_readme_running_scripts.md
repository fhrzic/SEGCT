# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"RunScripts"}}$ directory and basically everywhere. This readme is mostly connected to <span style="color:green;"> "setup.py"</span> and gives a brief overview of all executable scripts and how are they ment to be used. 

The scripts which can be exectued trough comand line with the brief descriptions what they are doing are as follows:

<ul>
<li><b>"split_data_from_xlsx"</b> -- <span style="color:green;"> "DatasetFingerprint.split_data_from_xlsx.py"</span> -- <b>01_readme_fingerprint</b> -- generating train, test, valid directories. </li>
<hr>
<li><b>"split_data_from_ratio"</b> -- <span style="color:green;"> "DatasetFingerprint.split_data_from_ratio.py"</span> -- <b>01_readme_fingerprint</b> -- generating train, test, valid directories.</li>
<hr>
<li><b>"generate_dataset_fingerprint"</b> -- <span style="color:green;"> "DatasetFingerprint.generate_dataset_fingerprint.py"</span> -- <b>01_readme_fingerprint</b> -- generating fingerprint (statistics) of data.</li>
<hr>
<li><b>"dataset_formating"</b> -- <span style="color:green;"> "DatasetPreprocessing.dataset_formating.py"</span> -- <b>02_readme_preprocessing</b> -- data normalization based on provided directories.</li>
<hr>
<li><b>"create_dataset"</b> -- <span style="color:green;"> "DatasetPreprocessing.create_dataset.py"</span> -- <b>02_readme_preprocessing</b> -- pipeline including dataset splitting + fingerprint + normalization.</li>
<hr>
<li><b>"verify_dataloader"</b> -- <span style="color:green;"> "Data.verify_dataloader.py"</span> -- <b>04_readme_dataloader</b> -- script for verifying dataset, dataloader and sampler.</li>
<hr>
<li><b>"inspect_model"</b> -- <span style="color:green;"> "Models.inspect_model.py"</span> -- <b>05_readme_models</b> --model inspection.</li>
<hr>
<li><b>"inspect_loss"</b> -- <span style="color:green;"> "Losses.inspect_loss.py"</span> -- <b>06_readme_losses</b> -- losses inspection.</li>
<hr>
<li><b>"test_training"</b> -- <span style="color:green;"> "Training.manual_training.py"</span> -- <b>08_readme_training</b> -- run training from manual configuration in script.</li>
<hr>
<li><b>"run_training_from_config"</b> -- <span style="color:green;"> "Training.configuration_based_training.py"</span> -- <b>08_readme_training</b> -- run training based on the provided config files. User modifies jsons.</li>
<hr>
<li><b>"run_inference"</b> -- <span style="color:green;"> "Inference.perform_inference.py"</span> -- <b>09_readme_inference</b> -- inference run based on provided model and data.</li>
<hr>
<li><b>"archive_model"</b> -- <span style="color:green;"> "Inference.archive_model.py"</span> -- <b>09_readme_inference</b> -- transfers all necessary files from trained model to the TraineModelsArchive directory</li>
<hr>
<li><b>"run_inference_from_dir"</b> -- <span style="color:green;"> "Inference.run_inference_from_dir.py"</span> -- <b>09_readme_inference</b> -- inference run based on provided data and output dir of a model stored in TrainedModelsArchive.</li>
<hr>
<li><b>"run_statistics"</b> -- <span style="color:green;"> "Inference.inference_statistics_evaluation.py"</span> -- <b>09_readme_inference</b> -- statistics evaluation based on caluclated inferences</li>
<hr>
<li><b>"run_preset_pipeline"</b> -- <span style="color:green;"> "RunScripts.run_preset_pipeline.py"</span> -- <b>00_readme_running_scripts</b> -- run preset pipline which includes dataset splitting + fingerprint + preprocessing + model training</li>
<hr>
<li><b>"run_argument_pipeline"</b> -- <span style="color:green;"> "RunScripts.run_argumented_pipeline.py"</span> -- <b>00_readme_running_scripts</b> -- run preset pipline which includes dataset splitting + fingerprint + preprocessing + model training</li>
<hr>

