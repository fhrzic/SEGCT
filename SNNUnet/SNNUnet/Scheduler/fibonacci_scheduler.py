""" Implemntation of novel fibonacciLnLRScheduler """
from typing import List
import numpy as np
import sympy as sp
from torch.optim.lr_scheduler import _LRScheduler

class FibonacciLnLRScheduler(_LRScheduler):
    """
    Class which implements novel fibonacciLnLRSScheduler which is advanced
    version of step scheduler. It creates arches where every arch is negative
    ln function with given constraints. Arches are long k epochs where k is based on
    fibonacci sequence (1,1,2,3,5...). k=1 represents n epochs where n is given as a 
    constraint which contains at least p_arch percent of max epochs. 
    
    For instance
    if max epochs is set to be 1000, and p_arch is set to 10% which means that arch
    must have at least 0.1*1000=100 epochs. Last sequence which satisfies this constraint
    is 0,1,1,2,3 because 0+1+1+2+3 = 7, 1000/7 ~ 143. In case that we get one more fibo seqence
    element 0+1+1+2+3+5 = 12, 1000/12 ~ 83 --> fails constraint of 10% arch lenght. 

    Each arch is mimicing function: f(x)= a * (-np.log(x + b) + c where a is fixed to 1.0, b and c
    are calculated such that the function satisfies following constraint f(0) = 5, f(arch_end) = 0.
    The beggining and end learning rates of the arch are then mapped to the function (simple 
    proportion mapping)

    The beggining and end learning rates of each arch are linearly caluclated. E.G. if begining lr 
    is 1e-2 and last one is 0 (always is last set to 0), and we have 5 arches (0-1,1-1, 1-2, 2-3) 
    lrs are as follows [0.01   0.0075 0.005  0.0025 0.]. Arch lengts in epochs 
    [0, 142, 284, 568, 1000] --> first arch is from 0 to 142, second is 142-284, third is 284-256, 
    and last one is 568-1000.
    """

    def __init__(
        self,
        optimizer,
        initial_lr: float = 1e-2,
        max_epochs: int = 1000,
        arch_p: float = 0.1,
        last_epoch: int = -1):
        """
        Init function of scheduler

        Args:
            * optimizer, member of torch.optim class, optimizer which is going to be used to
            train the model.
            * max_epochs, int, number of total epochs for which the model will be trained.
            * arch_p, float, minimum required number of epochs in every arch calculated as
            arch_p*max_epochs.
            * lasat_epoch, in case that we are resuming the training

        """
        # Set max epochs and other attributes
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr
        self.arch_p = arch_p

        # Obtain learning rates
        _lrs_class = LearningRateValueEstimation(number_of_epochs = max_epochs,
                                                    arch_p= arch_p,
                                                    init_learning_rate = initial_lr)
        self.learning_rates = _lrs_class.obtain_learning_rates()

        # Inherit
        super().__init__(optimizer, last_epoch)

        # Transfer learning rates to the optimizer (this is basically initialisation)
        for _param_group in optimizer.param_groups:
            _param_group['lr'] = self.learning_rates[0]

        # Set initial epoch
        self.last_epoch = -1

    def get_lr(
        self):
        """
        Method which calculates the learning rate based on the current epoch
        """
        # Handle initial learning rate
        if self.last_epoch == 0:  # Ensure the learning rate for epoch 0 is the initial value
            return [self.initial_lr for _ in self.optimizer.param_groups]

        # Safety meassure
        _current_index = min(self.last_epoch, len(self.learning_rates) - 1)

        # Return
        return [self.learning_rates[_current_index] for _ in self.optimizer.param_groups]
    

    def state_dict(
        self)->dict:
        """
        Return the state of the scheduler.
        """
        return {
            'initial_lr': self.initial_lr,
            'max_epochs': self.max_epochs,
            'arch_p': self.arch_p,
            'learning_rates': self.learning_rates,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(
        self, 
        state_dict: dict = None):
        """
        Load the scheduler state.

        Args:
            * state_dict, dict, dictionary containing the state of the scheduler.
        """
        self.initial_lr = state_dict['initial_lr']
        self.max_epochs = state_dict['max_epochs']
        self.arch_p = state_dict['arch_p']
        self.learning_rates = state_dict['learning_rates']
        self.last_epoch = state_dict['last_epoch']

        # Re-initialize the optimizer learning rates with the loaded learning rates
        for _param_group in self.optimizer.param_groups:
            _param_group['lr'] = self.learning_rates[0]

        # Call the super class load_state_dict if necessary
        super().load_state_dict(state_dict)
    

class LearningRateValueEstimation:
    """
    Class which helps to create learning rate scheduler in form of two arches
    between two points. It is like cascades where the total number of epochs
    is splitted in segments based on Fibonacci sequence with the smallest cascade having at least
    p% of epochs in the arch.
    """
    def __init__(
        self,
        number_of_epochs: int = 1000,
        arch_p:float = 0.1,
        init_learning_rate: float = 1e-2):
        """
        Init method which builds the main function for the cascade learning rate scheduler.
    
        Args:
            * number_of_epochs, int, number of epochs for which we need to calculate the learning rates
            * arch_p, float, percentage of total epochs which must be the minimal number of epochs
            in a single arch. E.G. number_of_epochs = 1000, arch_p = 0.1, minimum number of epochs in 
            arch must be 100.
            * init_learning_rate, float, intial learning rate - nothing fancy here.
        """
        # Calculate minimum number of samples in arch
        _min_epochs_in_arch = int(arch_p * number_of_epochs)
    
        # Obtain epoch numbers where the arches end    
        self.epochs = self.__fibo_sequence(
            min_epochs_in_arch = _min_epochs_in_arch,
            number_of_epochs = number_of_epochs)

        # Obtain learning rates at each epoch
        self.interim_lrs = self.__obtain_interim_learning_rates(
            init_lr = init_learning_rate,
            epochs = self.epochs)

    def __fibo_sequence(
        self,
        min_epochs_in_arch: int = 0,
        number_of_epochs:int = 0)->List[int]:
        """
        Method which generates fibonnachi sequence and returns first n 
        elements of it

        Args:
            * min_epochs_in_arch, int, minimum number of epoch which needs to be in arch
            * number_of_epochs, int, total number of epochs
        Return:
            * list of the first n elements of the sequence
        """
        # Find number of fibonachi elements such that minimum number epochs per archs is
        # greater than min_epoch_in_arch
        # First two elements are hardcoded
        _fib_sequence = [0, 1]  # Starting values of the Fibonacci sequence

        # Initial number of samples per arch
        _number_of_samples_per_arch = number_of_epochs / np.sum(_fib_sequence)
        # Iterate and seek for highest number of elements when the constraint is satisfied
        while _number_of_samples_per_arch > min_epochs_in_arch:
            _fib_sequence.append(_fib_sequence[-1] + _fib_sequence[-2])
            _number_of_samples_per_arch = number_of_epochs / np.sum(_fib_sequence)

        # One step before where constraint is still valid
        _fib_sequence = _fib_sequence[:-1]
        _number_of_samples_per_arch = number_of_epochs / np.sum(_fib_sequence)

        # Transfer elements of fibonacchi to epoch values
        _epoch_numbers = [0]
        for _idx in range(1, len(_fib_sequence)):
            _epoch_numbers.append(_fib_sequence[_idx] * int(_number_of_samples_per_arch) + _epoch_numbers[_idx-1])
        
        # Last number always must be max epoch (rounding)
        _epoch_numbers[-1] = number_of_epochs

        # Reverse it
        #_epoch_numbers = [number_of_epochs - _e for _e in _epoch_numbers]
        #_epoch_numbers.reverse()

        # Return
        return _epoch_numbers

    def __obtain_interim_learning_rates(
        self,
        init_lr:float = 1e-2,
        epochs: List[int] = None)->List[float]:
        """
        Method which calculate learning rate at each stage of the fibo
        sequence. In another words - it caluclates where each arch begin and
        end.

        Args:
            * init_lr, float, beggining learning rate while the ending learning rate is 
            0
            * epochs, list of integers, list to know how many arches are there
        """
        # Number of arches
        _num_arches = len(epochs)

        # Calculate interim learning rates
        _lr = np.linspace(init_lr, 0, _num_arches)

        # Return
        return _lr

    def rolemodel_function(
        self,
        number_of_samples: int = 100,
        beggining_value: float = 1e-3,
        end_value: float = 1e-2):
        """
        This is the rolemodel function which will be in every arch. This can be overriden by
        any function. Function implemented here is:
            
            f(x) = a * (-np.log(x + b) + c

        This function will be set in a way that it passes through points (0,5) and (number_of_samples, 0). Based
        on this the params b and d for the functions will be caluclated while a is fixed to 0.1.
        Afterwards the values will be mapped to interval beggining_value - end_value.
        
        Args:
            * number_of_samples,int, number of samples between beggining_value and end_value
            * beggining_value, float, desired value of the function in the beggining (x=0)
            * end_value, float, desired value of the function at the end of of interval (x=number_of_samples)

        Returns:
            * list of values from the interval mapped in the provided interval
        """
        # Define mandatory points necessary for function creation
        _y0 = 5
        _y1 = 0
        _x0 = 0
        _x1 = number_of_samples

        # Solve equation
        _a = 1.0
        _b, _c = sp.symbols('_b _c')
        _eq1 = _a * (-sp.log(_x0 + _b)) + _c - _y0 
        _eq2 = _a * (-sp.log(_x1 + _b)) + _c - _y1  

        # Solution
        _solution = sp.solve([_eq1, _eq2], (_b, _c))
        _b, _c = _solution[0]
        _b, _c = float(_b), float(_c)

        # Obtain function
        _f = lambda x, a, b, c: a * (-np.log(x + b)) + c

        # Real range - must be 1 to avoid log(0)
        _epochs = np.linspace(0, number_of_samples, number_of_samples, dtype=int)       

        # Obtain function values
        _function_values = [_f(_v, _a, _b, _c) for _v in _epochs]

        # Fix last value to be exact --> rounding errors of float
        _function_values[-1] = _y1

        # Map values
        _mapped_values = [beggining_value + ((end_value - beggining_value) / 
                                             (_function_values[-1] - _function_values[0])) * 
                          (x_i - _function_values[0]) for x_i in _function_values]
        # Fix last value --> rounding errors of float
        _mapped_values[-1] = end_value
        
        # Return values
        return _mapped_values

    def obtain_learning_rates(
        self)->List[float]:
        """
        Main method for obtaining learning rates
        
        Returns:
            * list of floats which are learning rates per each epoch for given
            epoch range in the init function
        """
        # Make storage
        _learning_rates = []

        # Obtain samples
        for _i in range(len(self.epochs)-1):
            _arch_lr = self.rolemodel_function(
                number_of_samples = self.epochs[_i+1] - self.epochs[_i],
                beggining_value = self.interim_lrs[_i], 
                end_value = self.interim_lrs[_i+1])
            _learning_rates.extend(_arch_lr) 
        # Return
        return(_learning_rates)
