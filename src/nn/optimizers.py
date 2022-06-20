"""Optimizer Related Utilities"""
import typing
from typing import List

import bitsandbytes as bnb
from torch import nn
from torch.optim import Optimizer

__all__ = ["fetch_optimizer"]


@typing.no_type_check
def fetch_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimizer: str = "AdamW",
) -> Optimizer:
    """
    Get the Optimizer

    :param model: Model
    :type model: nn.Module
    :param learning_rate: Learning Rate
    :type learning_rate: float
    :param weight_decay: Weight Decay
    :type weight_decay: float
    :param optimizer: which optimizer to use, defaults to "AdamW"
    :type optimizer: str, optional
    :return: Optimizer with weight decay
    :rtype: Optimizer
    """

    param_optimizer: List = list(model.named_parameters())
    no_decay: List = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer == "AdamW":
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return optimizer
