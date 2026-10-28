# Imports
import sys, os
import torch
#from torch.cuda.amp import GradScaler
from torch.nn import NLLLoss, CrossEntropyLoss
from tqdm import tqdm
import shutil
from accelerate import Accelerator
import json


# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

# This imports can be changed based on the preferences!
from Data.Dataloader.dataloader import *
from Configurations.config_model import *
from Configurations.config_training import *
from Models.SNNUnet.SNNUnet import *
from Models.SNNUnet.autograd_and_gradient_clipping import *
from Losses.deep_supervision_loss import *
from Losses.compound_loss_DC_and_CE import *
from Scheduler.fibonacci_scheduler import *
from Scheduler.polylr import *
#*******************************************************************************
# Main training class
#*******************************************************************************
class MainTrainingClass():
    """
    Class for main training script
    """
    def __init__(self, 
                 train_dataloader: DataLoader = None,
                 valid_dataloader: DataLoader = None,
                 model_config: ModelConfig = None,
                 training_config: TrainingConfig = None):
        """
        Init method for training. It requires dataloader of training data, model_config 
        which is a config file for a model which is going to be used. Also it requires 
        training_config which specifies all training configurations.

        Args:
            * dataloader, torch.Dataloader. On this dataloader all the training will be done
            * model_config, modelConfig, config file for the model. Based on this config model 
            will be build
            * training_config, trainingConfig, training config which specifies how the model 
            will be trained.
        """

        # Check if all necessary config files are provided
        assert train_dataloader is not None and \
            valid_dataloader is not None and \
            model_config is not None and \
            training_config is not None,\
            "Some of config files or dataloader is missing. Please specify them!"

        # Make config available
        self.training_config = training_config

        # Assign dataloader
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # Build model
        self.model_config = model_config
        self.model = self.__obtain_model(
            model_config = self.model_config,
            device = self.training_config.device)

        # Obtain loss
        self.training_loss, self.valid_loss = self.__obtain_loss(
            training_config = self.training_config)

        # Obtain optimizer
        self.optimizer, self.lr_scheduler = self.__obtain_optimizer(
            training_config = self.training_config)
            
        # Set gradient scaler
        if self.training_config.gradient_scaler:
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = None

        # Create output dir for results
        if os.path.exists(training_config.output_dir):
            shutil.rmtree(training_config.output_dir)
        os.makedirs(training_config.output_dir)
        

    # Model definition
    def __obtain_model(self, model_config: ModelConfig = None, device: str = "cpu")->torch.nn.Module:
        """
        Method which configures and create model

        Args:
            * model_config, modelConfig, the user is in complete response to provide 
            valid model configuration
            * device, string, string describing where the model will be set
        
        Returns:
            * model obtained by torch.nn.Module
        """
        # Check for the implementation of the model
        assert model_config.model_name in ["VNet", "SNNUnet"],\
            f"Unknown model {model_config.model_name}. Please provide valid model!"

        # VNet
        if model_config.model_name == "VNet":
            _model = VNet(model_config)
        
        if model_config.model_name == "SNNUnet":
            _model = SNNUnet(**dict(model_config))
        
        # Transfer
        _model = _model.to(device)

        return _model

    # Optimizer definition
    def __obtain_optimizer(
        self,
        training_config: TrainingConfig = None)->list:
        """
        Method for obtaining the optimizer.

        Args:
            * training_config, trainingConfig, eseentially, this contains all neccessary 
            data for building optimizer and learning rate scheduler

        Returns:
            * optimizer and learning rate schiduler objects
        """
        assert training_config.optimizer_params.optimizer_name == "SGD",\
            f"Unknown optimizer {training_config.optimizer_params.optimizer_name}! Please provide valid optimizer for training"
        # Obtain optimizer
        if training_config.optimizer_params.optimizer_name == "SGD":
            _optimizer = torch.optim.SGD(
                self.model.parameters(), 
                training_config.optimizer_params.learning_rate,
                weight_decay = training_config.optimizer_params.weight_decay,
                momentum=training_config.optimizer_params.momentum,
                nesterov=training_config.optimizer_params.nesterov)
        
        # Obtain scheduler
        if training_config.optimizer_params.scheduler_name == None:
            return _optimizer, None
        assert training_config.optimizer_params.scheduler_name in ["poly", "fibo"],\
            f"Unknown scheduler {training_config.optimizer_params.scheduler_name}! Please provide valid scheduler for training"
        if training_config.optimizer_params.scheduler_name == "poly":
            _lr_scheduler = PolyLRScheduler(
                    _optimizer,
                    training_config.optimizer_params.learning_rate,
                    training_config.epochs)
        if training_config.optimizer_params.scheduler_name == "fibo":
            _lr_scheduler = FibonacciLnLRScheduler(
                    _optimizer,
                    initial_lr = training_config.optimizer_params.learning_rate,
                    max_epochs = training_config.epochs,
                    arch_p = 0.2)
        # Return optimizer and scheduler
        return _optimizer, _lr_scheduler

    # Obtain Loss
    def __obtain_loss(self, training_config: TrainingConfig = None)->list:
        """
        Method for obtaining the loss for the training. It returns loss function

        Args:
            * training_config, which is actually training config with training_loss_function and 
            valid_loss_function.
        
        Returns:
            * two loss functions, first one is for training and second one is for validaiton
        """
        ## SOLVE TRAINING LOSS
        # Check if there is a complex loss
        if isinstance(training_config.training_loss_function, str):
            # It is a simple loss
            assert training_config.training_loss_function in ["nll", "cross_entropy"],\
                f"Unkown loss function {training_config.training_loss_function}. Please check the loss function"
            # Set nll loss
            if training_config.training_loss_function == "nll":
                _training_loss = NLLLoss()
            # Cross entropy
            if training_config.training_loss_function == "cross_entropy":
                _training_loss = CrossEntropyLoss()
        else:
            # Figure out which loss - It is complex one
            assert training_config.training_loss_function.loss_name in ["DCandCE", "dsLoss" ],\
                f"Unkown loss function {training_config.training_loss_function.loss_name}. Please check the loss function"
            # DCandCE
            if training_config.training_loss_function.loss_name == "DCandCE":
                _training_loss = DC_and_CE_loss(config = training_config.training_loss_function)

            # Deep supervision loss
            # DCandCE
            if training_config.training_loss_function.loss_name == "dsLoss":
                _training_loss = DeepSupervisionLoss(
                    ds_config = training_config.training_loss_function)

        ## SOLVE VALIDATION LOSS (It does not support deep supervision)
        # Check if there is a complex loss
        if isinstance(training_config.validation_loss_function, str):
            # It is a simple loss
            assert training_config.validation_loss_function in ["nll", "cross_entropy"],\
                f"Unkown loss function {training_config.validation_loss_function}. Please check the loss function"
            # Set nll loss
            if training_config.validation_loss_function == "nll":
                _validation_loss = NLLLoss()
            # Cross entropy
            if training_config.validation_loss_function == "cross_entropy":
                _validation_loss = CrossEntropyLoss()
        else:
            # Figure out which loss - It is complex one
            assert training_config.validation_loss_function.loss_name in ["DCandCE"],\
                f"Unkown loss function {training_config.validation_loss_function.loss_name}. Please check the loss function"
            # DCandCE
            if training_config.validation_loss_function.loss_name == "DCandCE":
                _validation_loss = DC_and_CE_loss(config = training_config.validation_loss_function)

        # Return
        return _training_loss, _validation_loss

    # Train one epoch
    def train_one_epoch(self):
        """
        Method which trains for one epoch.

        Args: 
            * epoch_num, int, cosmetic to show 

        """
        # Swap to mode train
        self.model.train()

        # Create loss storage
        _loss_storage = torch.zeros(
            self.training_config.batch_per_epoch, 
            device = self.training_config.device)
        # Create loading bar
        _pbar = tqdm(total=self.training_config.batch_per_epoch, desc="Epoch")

         #Iterate over batch
        for _i, _batch in enumerate(self.train_dataloader):
            # Set optimizer
            self.optimizer.zero_grad(set_to_none = True)

            # Obtain data from batch
            _ct_patch, _seg_mask_patch, _auxiliary_masks, \
                _random_idx, _patch_location, _fg, _class = _batch
            
            # Transfer to device
            if self.training_config.accelerator:
                _ct_patch = _ct_patch.to(self.accelerator.device)
                _seg_mask_patch = _seg_mask_patch.to(self.accelerator.device)
                for _idx, _ in enumerate(_auxiliary_masks):
                    _auxiliary_masks[_idx] = _auxiliary_masks[_idx].to(self.accelerator.device)
            else:
                # Classical transfer
                _ct_patch = _ct_patch.to(self.training_config.device)
                _seg_mask_patch = _seg_mask_patch.to(self.training_config.device)
                for _idx, _ in enumerate(_auxiliary_masks):
                    _auxiliary_masks[_idx] = _auxiliary_masks[_idx].to(self.training_config.device)

            # Make prediction and obtain loss. "I stole" this from nnUnet trianing. They claim
            # it is necessary if training on CPU. Well, I have not test it to on CPU - Yet
            # Dummy context is there for it. Also this will happen only if mixed_precision
            # is set to true.
            with torch.autocast(device_type=_ct_patch.device.type, enabled=True) \
                if _ct_patch.device.type == 'cuda' and self.training_config.mixed_precision else DummyContext():
                # Obtain predictions
                _prediction = self.model(_ct_patch)
        
                # Fix output
                if self.training_config.training_loss_function == 'nll':
                    _seg_mask_patch = _seg_mask_patch.view(_seg_mask_patch.numel())

                # Calculate loss
                if _auxiliary_masks:
                    # This means that we have auxiliary outputs and that we
                    # need to provide them to the loss function
                    _loss = self.training_loss(_prediction, _auxiliary_masks)
                else:
                    # No deep supervision or auxiliary outputs, standard output goes to the
                    # loss function. SNNUnet is having list output as default
                    _loss = self.training_loss(_prediction, _seg_mask_patch)

            # Backpropagate loss
            if self.grad_scaler is not None:
                # Well we can't have accelerate and grad scaler
                if self.training_config.accelerator:
                    self.accelerator.backward(_loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    self.optimizer.step()
                else:
                    self.grad_scaler.scale(_loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
            else:
                # No grad scaller applies
                if self.training_config.accelerator:
                    self.accelerator.backward(_loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    self.optimizer.step()
                else:
                    _loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                    self.optimizer.step()
            # Update progress bar
            _pbar.update(1)

            # Save loss
            _loss_storage[_i] = _loss.detach()

            # Break when needed - this is inifinte loader
            if _i == self.training_config.batch_per_epoch-1:
                break
        _pbar.close()

        return _loss_storage

    def validate_one_epoch(self):
        """
        Due to its similarity, we are going to validate "one epoch" of valid dataloader
    
        Returns:
         * Return dictionary holding values for: Precision, Recall, F1-Score, and Loss as averages of all samples in validation/test
        """
        # Swap to mode evaluate
        self.model.eval()
        # Create loading bar
        _pbar = tqdm(total = len(self.valid_dataloader.dataset), desc="Validatation")
        # We don't need calculate gradients
        with torch.no_grad():
            # Get overal metrices
            _overall_precission = []
            _overall_recalls = []
            _overall_f1 = []
            _overall_loss = []
            # Go trough data
            for _index, _batch in enumerate(self.valid_dataloader):
                # Create tmp storage
                _tmp_predictions = []
                _tmp_true = []
                _tmp_loss = []

                # Obtain data
                _ct, _seg = _batch

                # Reshape
                _ct = _ct.squeeze(0)
                _seg = _seg.squeeze(0)
                _ct = _ct.unsqueeze(1)
                _seg = _seg.unsqueeze(1)

                _ct = _ct.to(self.training_config.device)
                _seg = _seg.to(self.training_config.device)

                # Make smaller batches
                _ct_splits = torch.split(_ct, self.training_config.batch_size, dim=0)
                _seg_splits = torch.split(_seg, self.training_config.batch_size, dim=0)

                # Obtain predictions for a given case
                for _split_ct, _split_seg in zip(_ct_splits, _seg_splits):
                    # Transfer to device
                    if self.training_config.accelerator:
                        _split_ct = _split_ct.to(self.accelerator.device)
                        _split_seg = _split_seg.to(self.accelerator.device)
                    else:
                        _split_ct = _split_ct.to(self.training_config.device)
                        _split_seg = _split_seg.to(self.training_config.device)

                    # Apply mixed precision if necessary
                    with torch.autocast(device_type=_split_ct.device.type) \
                        if _split_ct.device.type == 'cuda' and self.training_config.mixed_precision else DummyContext():
                        # Obtain prediction
                        _prediction = self.model(_split_ct)

                        # Check for deep supervision
                        if isinstance(_prediction, list):
                            _prediction = _prediction[-1]

                        # Get loss and append it to losses
                        _tmp_loss.append(self.valid_loss(_prediction, _split_seg.squeeze(1)).to("cpu"))

                        # Obtain label
                        _prediction = torch.argmax(_prediction, dim = 1)

                        # Fix output
                        #if self.training_config.loss_function == 'nll':
                        #    _split_seg = _split_seg.view(_split_seg.numel())

                        # DEBUG
                        _tmp_predictions.append(_prediction)

                # CALCULATING METRICS
                # Concatenated outputs
                _tmp_predictions = torch.cat(_tmp_predictions, dim = 0)
                #_tmp_true = torch.cat(_tmp_true, dim = 0)
                _tmp_true = _seg.squeeze(1)
                # Calculate precision and recal
                _precisions, _recalls = [], []

                # Calculate losses
                _losses = torch.stack(_tmp_loss).mean(dim=0).cpu()
                _overall_loss.append(_losses)
                # I am skipping background
                for _cls in range(1, self.model_config.number_of_output_classes):
                    _pred_mask = (_tmp_predictions == _cls)
                    _label_mask = (_tmp_true == _cls)
                    _tp = (_pred_mask & _label_mask).float().sum()
                    _fp = (_pred_mask & ~_label_mask).float().sum()
                    _fn = (~_pred_mask & _label_mask).float().sum()
                    
                    _precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else float('nan')
                    _recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else float('nan')
                    
                    _precisions.append(_precision)
                    _recalls.append(_recall)
                
                # Calculate mean precission and recall
                _mean_precision = torch.nanmean(torch.tensor(_precisions))
                _mean_recall = torch.nanmean(torch.tensor(_recalls))
                
                # F1 score
                if _mean_recall + _mean_precision != 0:
                    _overall_f1.append(2*(_mean_precision*_mean_recall) / (_mean_precision + _mean_recall))

                # Update storage
                _overall_precission.append(_mean_precision)
                _overall_recalls.append(_mean_recall)

                # Update progress bar
                _pbar.update(1)
        _pbar.close()

        _mean_precision = float(torch.nanmean(torch.tensor(_overall_precission)))
        _mean_recall = float(torch.nanmean(torch.tensor(_overall_recalls)))
        _mean_f1 = float(torch.nanmean(torch.tensor(_overall_f1)))
        _mean_loss = float(torch.nanmean(torch.tensor(_overall_loss)))
        
        print(f"Validation: Precision = {_mean_precision}",
              f"Recall = {_mean_recall}",
              f"F1-Score = {_mean_f1}", 
              f"Loss = {_mean_loss}", 
              flush=True)

        # Return dict
        _return_dict = {"Precision": _mean_precision,
                        "Recall": _mean_recall,
                        "F1-Score": _mean_f1,
                        "Loss": _mean_loss}

        # Return
        return _return_dict

    def __save_model(
        self,
        epoch: int = 0,
        name: str = None):
        """
        Method for saving model. It saves full model as well as the checkpoint for 
        resuming model training. Checkpoint ( containing optimizer state, 
        lr rate etc...) is saved always under the same name (not for every)

        Args:
            * epoch, int, number of epoch
            * name, str, name must be cerefully defined because if it has "best_model" in 
            it, then the check point is also saved as well as standard check point (by check point
            I consider .pth which contains all states for training resume)
        """

        # Export model (name is given)
        torch.save(self.model, os.path.join(self.training_config.output_dir, name))

        # Now save the checkpoints
        _checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler!= None else None,
            'grad_scaler_state': self.grad_scaler.state_dict()\
                if self.grad_scaler is not None else None,
            'epoch': epoch
        }
        torch.save(_checkpoint_dict, 
                   os.path.join(self.training_config.output_dir, "last_checkpoint.pth"))

        # If it is a best checkpoint - we need to save additional best checkpoint
        if "best_model" in name:
            torch.save(_checkpoint_dict, 
                   os.path.join(self.training_config.output_dir, "best_model_checkpoint.pth"))

    def __load_model(
        self)->int:
        """
        Method which loads model from CHECKPOINT and continues to train the model until
        the last epoch. Also scheduler and optimizer states are also loaded.

        Returns:
            * int which represent the last epoch when the model was saved.
        """
        try:
            _checkpoint = torch.load(self.training_config.resume_from_checkpoint)
            self.model.load_state_dict(_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(_checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(_checkpoint['scheduler_state_dict'])
            self.grad_scaler.load_state_dict(_checkpoint['grad_scaler_state'])
            return _checkpoint['epoch']
        except ValueError:
            return 0

    # Main train script
    def start_training(self):
        """
        Main script for training
        """

        # Enable accelerator
        if self.training_config.accelerator:
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(
                self.train_dataloader,
                self.valid_dataloader)

        # Load checkpoint
        if self.training_config.resume_from_checkpoint is not None:
            _begin_epoch = self.__load_model()
        else:
            _begin_epoch = 0

        # Best_f1
        _best_f1 = 0
        # Storage dict
        _storage = {"Training_loss": [],
                    "Valid_loss": [],
                    "Valid_F1_Score": [],
                    "Valid_Precision": [],
                    "Valid_Recall": []}
        for _i in range(_begin_epoch, self.training_config.epochs):
            print("####################################################", flush=True)
            print(f"Working on epoch: {_i}", flush=True)
        
            # Train
            _loss = self.train_one_epoch()
            
            # Step scheduler
            self.model.train()
            if self.lr_scheduler != None:
                self.lr_scheduler.step(_i+1)

            # Report loss           
            print(f"Mean loss through this epoch was: {_loss.mean()}", flush=True)
            _storage["Training_loss"].append(_loss.mean().item())
        
            # Go through validation
            print("Working on validation...", flush=True)
            _val_metrics_dict = self.validate_one_epoch()
            _val_f1 = _val_metrics_dict["F1-Score"]
            _storage["Valid_loss"].append(_val_metrics_dict["Loss"])
            _storage["Valid_Precision"].append(_val_metrics_dict["Precision"])
            _storage["Valid_Recall"].append(_val_metrics_dict["Recall"])
            _storage["Valid_F1_Score"].append(_val_metrics_dict["F1-Score"])

            # Save model if best
            if _val_f1 > _best_f1:
                self.model.eval()
                self.__save_model(epoch = _i, name = "best_model.pth")
                print(f"Saved best model! {_best_f1}-->{_val_f1}", flush=True)
                _best_f1 = _val_f1
            
            # Save model if necessary
            if _i % self.training_config.save_model_interval == 0:
                self.model.eval()
                print(f"Checpoint model saving!", flush=True)
                self.__save_model(epoch = _i, name = f"epoch_{_i}.pth")
        
        # Save last model/epoch
        print(f"Checpoint model saving!", flush=True)
        self.__save_model(epoch= self.training_config.epochs, 
                          name = f"epoch_{self.training_config.epochs}.pth")

        # Export of losses to see convergnece
        with open(os.path.join(self.training_config.output_dir, f"training_log.json"), 'w') as _json_file:
            json.dump(_storage, _json_file, indent=4)

