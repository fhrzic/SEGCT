""" Scheduler tailored the same way as the NNUnet"""
from torch.optim.lr_scheduler import _LRScheduler

class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None):
        """
        Init of the class that implements LR Scheduler the same way
        it is implemented in the NNUnet.

        Args:
            * optimizer, optimizer which is going to be used. (SGD default)
            * initial_lr, float, initial learning rate at the beginning.
            * max_steps, int, number of maximum steps which optimizer can take.
            * exponent, float, necessary for the calculation of new learning rate.
            * current_step, int, current step typically epoch.
        """
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)
        
    def get_lr(self):
        """
        Return the calculated learning rates based on the current state.
        """
        # Determine the current step -1 idk but this mimics lr from the NNUnet scheduler
        # After 3h of doing it - I give up.
        current_step = self.last_epoch #-1 if self.last_epoch != -1 else self.ctr
        # Calculate the learning rate based on the current step
        factor = max(0.0, 1 - current_step / self.max_steps)  # Ensure no negative values
        new_lr = self.initial_lr * factor ** self.exponent
        return [new_lr for _ in self.optimizer.param_groups]

    def step(self, epoch = None):
        """
        Update the learning rate based on the current epoch or internal counter.
        """
        # Update the internal counter if no epoch is specified
        if epoch is None:
            epoch = self.ctr
            self.ctr += 1

        # Call the parent's step method with the correct epoch
        super().step()  # Update optimizer param groups with get_lr
        
        # Update epoch
        self.last_epoch = epoch

    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'initial_lr': self.initial_lr,
            'max_steps': self.max_steps,
            'exponent': self.exponent,
            'ctr': self.ctr
        }

    def load_state_dict(self, 
                        state_dict:dict = None):
        """
        Load the scheduler state.

        Args:
            * state_dict, dict, dictionary contating all necessary
            steps
        """
        self.initial_lr = state_dict['initial_lr']
        self.max_steps = state_dict['max_steps']
        self.exponent = state_dict['exponent']
        self.ctr = state_dict['ctr']
