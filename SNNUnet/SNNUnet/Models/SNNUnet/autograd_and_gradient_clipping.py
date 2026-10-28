"""
Autograd and gradient clipping needs some utility functions.
Nothing special, but it must be put somwhere, so I decided
to put it in separte script to keep training script clear
"""

import torch
from torch.amp.grad_scaler import OptState

__all__ = ["GradScaler", "OptState"]

class GradScaler(torch.amp.GradScaler):
    """
    This is well, I don't want to say what... I used src code:
    https://pytorch.org/docs/stable/_modules/torch/cpu/amp/grad_scaler.html#GradScaler

    To set GradScaler to cuda

    """
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            "cuda",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )

class DummyContext(object):
    """
    This was taken from the nnUnet. It is nice 
    workarround when there must still be contex provided
    but we have nothing to provide.
    """
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
