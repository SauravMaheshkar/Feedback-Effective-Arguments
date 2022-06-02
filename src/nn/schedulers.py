"""Scheduler related Utilities"""
import typing
from typing import Any

import torch
from torch.optim import lr_scheduler


@typing.no_type_check
def fetch_scheduler(
    scheduler: str,
    optimizer: torch.optim.Optimizer,
    min_lr: float,
    t_max: int,
    t_0: int,
) -> Any:
    """
    Returns a Scheduler based on the provided parameters
    :param scheduler: Which Scheduler to use
    :type scheduler: str
    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param min_lr: Minimum Learning Rate
    :type min_lr: float
    :param t_max: Maximum number of iterations
    :type t_max: int
    :param t_0: Number of iterations for the first restart
    :type t_0: int
    :return: Scheduler based on the provided parameters
    :rtype: Any
    """
    if scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=min_lr
        )
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, eta_min=min_lr
        )
    elif scheduler is None:
        return None

    return scheduler
