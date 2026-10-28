""" Dataloaders definitin"""
from typing import List, Tuple
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from Configurations.config_data import LoaderConfig

def infinite_dataloader(loader_config: LoaderConfig = None)->DataLoader:
    """
    Function for generating dataloader. It has custom collet function because of deep supervision.
    Namely, dataset can return list of tensors which bece list of list of tensors and those can
    not be stacked properly with regular collet function. 
    
    Args:
        * loader_config, loaderConfig, all necessary config properties to build data loader
        
    Returns:
        * Dataloader, torch dataloader ready for infinite sampling. 
        BE CAREFULL THIS THING SAMPLES INFINITLY
    """
    # Obtain dataloader
    _dataloader =  DataLoader(dataset = loader_config.dataset,
                              batch_sampler = loader_config.sampler,
                              collate_fn= loader_config.collate_fn,
                              num_workers = loader_config.number_of_workers,
                              pin_memory = loader_config.use_gpu,
                              pin_memory_device = loader_config.device)

    # Return
    return _dataloader

def custom_collate_fn_for_infinite_dataloader(
    batch_size: int = None,
    batch: list = None)->Tuple[torch.Tensor, torch.Tensor,
                              List[torch.Tensor], torch.Tensor,
                              torch.Tensor, bool, int]:
    """
    This is custom collate function for infinite dataloader. Namely the problem
    is that list of torch tensors for deeps supervision requires additional treatment.
    Normal colate function can not handle different tensor resolutions (to stack them in
    one tensor) and the number of tensors is unkown so we can not make a placeholders
    for the thensors

    Args:
        * batch_size, int, this is how collate_fn works (dataloader demands to give batch size)
        * batch, list, list of outputs from the __get__item of the provided dataset
        in the dataloader. Please see dataset.py __getitem__() method to see what is in 
        the batch
    
    Outputs:
        * batch, but all tensors are stacked. This means in case of deep supervision
        we will not have list of lists of torch tensors, but list of torch tenosors where
        the first dim will be the batch size. A lot of work for little consistency if
        you ask me.
    """
    # Create sotrages
    _ct_patch_storage = []
    _seg_mask_patch_storage = []
    _ds_mask_patches_storage = [[] for _ in range(len(batch[0][2]))]
    _random_index_storage = []
    _patch_location_storage = []
    _foreground_storage = []
    _class_storage = []

    # Collect everything in list
    for _item in batch:
        _ct_patch_storage.append(_item[0])
        _seg_mask_patch_storage.append(_item[1])
        _random_index_storage.append(_item[3])
        _patch_location_storage.append(_item[4])
        _foreground_storage.append(_item[5])
        _class_storage.append(_item[6])

        # Special handeling of ds masks
        for  _idx, _ds_mask in enumerate(_item[2]):
            _ds_mask_patches_storage[_idx].append(_ds_mask)

    # Now stack tensors which can be stacked (class and foreground can not)
    _ct_patch_storage = torch.stack(_ct_patch_storage)
    _seg_mask_patch_storage = torch.stack(_seg_mask_patch_storage)
    _random_index_storage = torch.stack(_random_index_storage)
    _patch_location_storage = torch.stack(_patch_location_storage)
    # Special handeling of ds masks
    for _idx, _ds_mask in enumerate(_ds_mask_patches_storage):
        _ds_mask_patches_storage[_idx] = torch.stack(_ds_mask_patches_storage[_idx])

    # Return new stacked batch (same order)
    return [_ct_patch_storage, _seg_mask_patch_storage,
            _ds_mask_patches_storage, _random_index_storage,
            _patch_location_storage, _foreground_storage,
            _class_storage]


def one_sample_dataloader(loader_config: LoaderConfig = None)->DataLoader:
    """
    Function for generating dataloader. Just for the consistency reasons. This 
    dataloader returns one sample at the time which is convinient for validation and
    test sets.
    
    Args:
        * loader_config, loaderConfig, all necessary config properties to build data loader
        
    Returns:
        * Dataloader, torch dataloader generateing one sample at the time from the set
    """
    # Obtain dataloader
    _dataloader =  DataLoader(dataset = loader_config.dataset,
                              batch_size = 1,
                              num_workers = loader_config.number_of_workers,
                              pin_memory = loader_config.use_gpu,
                              pin_memory_device = loader_config.device)

    # Return
    return _dataloader

def custom_collate_fn_for_sampler_for_validation(
        batch_size: int = None,
        batch: list = None)->Tuple[torch.Tensor, 
                               torch.Tensor]:
    """
    This is custom collate function for validation sampler. Namely the problem
    is that the one_sampler_dataloader works in a way that it returns all the samples for
    observed case like a one big batch. And then the validation function in training is 
    made to "dismantle" this batch into smaller batches. To follow this logic,
    sampler_for_validation samples one case and number of samples contaning only foreground
    patches. From the whole batches, we need to obtain only patches and segmentation maps.

    Args:
        * batch_size, int, this is how collate_fn works (dataloader demands to give batch size)
        * batch, list, list of outputs from the __get__item of the provided dataset
        in the dataloader. Please see dataset.py __getitem__() method to see what is in 
        the batch
    
    Outputs:
        * batch, but all tensors are stacked which is compatible with the train_one_epoch method
        inside training_class. (Training/training_class.py)
    """

    # Create sotrages
    _ct_patch_storage = []
    _seg_mask_patch_storage = []
    
    # Collect everything in list
    for _item in batch:
        _ct_patch_storage.append(_item[0])
        _seg_mask_patch_storage.append(_item[1])
        
    # Now stack tensors 
    _ct_patch_storage = torch.stack(_ct_patch_storage)
    _seg_mask_patch_storage = torch.stack(_seg_mask_patch_storage)
    
    # Permute/fix channels for compatibility of training method in training_class.py
    _seg_mask_patch_storage = _seg_mask_patch_storage.unsqueeze(0)
    _ct_patch_storage = _ct_patch_storage.permute(1, 0, 2, 3, 4)
    #_seg_mask_patch_storage = _seg_mask_patch_storage.permute(1, 0, 2, 3, 4)

    # Return new stacked batch (same order)
    return [_ct_patch_storage, 
            _seg_mask_patch_storage,]