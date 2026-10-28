"""
To make things easier, if you want to implement
the sampler, here is a blueprint of a class for it.
Nothing special, just to make things easier to follow.

THIS IS NOT CALLED ANYWHERE! It serves like an overview!
"""

from abc import ABC, abstractmethod

# Abstract Class Definition
class SamplerExample(ABC):
    """
    Sampler is in charge of selecting the samples and implementing
    the logic behind it. For instance, we can choose which classes 
    are going to be sampled, how many foreground samples must be present
    in the batch, or literaly anything we desire.
    """

    @abstractmethod
    def __init__(self):
        """
        It must have init where we obtain typically some type of config which
        defines input dataset which can be prompted for the samples, ratios, 
        number of samples, batch size etc.
        """
        pass  # No implementation; subclasses must override it
    
    @abstractmethod
    def __iter__(self):
        """
        In this class we are implementing all "logic" behind the sampling.
        As a result, it returns batch of samples. Typically in my logic
        this method yield batches and the "stop" criterium is implemented
        in a for loop during the training.
        """
        pass

    @abstractmethod
    def __len__(self): 
        """
        It return number of batches generated from the dataset.
        Neat thing to have. In case of infinite dataloader,
        it returns infinity
        """
        pass