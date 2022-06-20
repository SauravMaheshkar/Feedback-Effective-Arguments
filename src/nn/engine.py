"""Training Engine"""
import copy
import gc
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Tuple

import numpy as np
import torch
import wandb
from rich import print
from rich.progress import track
from torch import nn

__all__ = ["run_training"]

##### Loss Function #####
def criterion(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the loss

    :param outputs: Output Logits from the Model
    :type outputs: torch.Tensor
    :param labels: Ground Truth labels
    :type labels: torch.Tensor
    :return: Loss
    :rtype: torch.Tensor
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def train_one_epoch(
    cfg: Dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    dataloader: Iterable,
    device,
    epoch: int,
    fold: int,
) -> float:
    """
    Train the model for One Epoch

    :param cfg: Configuration Dictionary
    :type cfg: Dict
    :param model: PyTorch Model
    :type model: nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Scheduler
    :param dataloader: Training Dataloader
    :type dataloader: Iterable
    :param device: Pytorch Device
    :param epoch: Epoch No. (for logging)
    :type epoch: int
    :param fold: Fold No. (for logging)
    :type fold: int
    :return: Loss after Epoch
    :rtype: float
    """

    # Put the model into train mode
    model.train()

    # Miscellaneous variables
    dataset_size = 0
    running_loss = 0.0
    iters = len(dataloader)  # type: ignore

    # Iterate over dataloader
    for step, data in track(enumerate(dataloader), total=iters):
        # Move tensors to device
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.long)

        # Get Batch Size
        batch_size = ids.size(0)

        with torch.cuda.amp.autocast():
            # Forward Pass
            outputs = model(ids, mask)

            # Calculate Loss and Perform Backpropagation
            loss = criterion(outputs, targets)

        loss = loss / cfg["n_accumulate"]
        scaler.scale(loss).backward()

        if (step + 1) % cfg["n_accumulate"] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                if cfg["scheduler"] == "CosineAnnealingWarmRestarts":
                    scheduler.step(epoch + step / iters)
                else:
                    scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # Log the metrics
        wandb.log({f"Train/Fold-{fold} Loss": epoch_loss})
        wandb.log({f"Train/Fold-{fold} LR": optimizer.param_groups[0]["lr"]})

    # Garbage Collection
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model: nn.Module, dataloader: Iterable, device, fold: int) -> float:
    """
    Perform Validation for One Epoch

    :param model: PyTorch Model
    :type model: nn.Module
    :param dataloader: Training Dataloader
    :type dataloader: Iterable
    :param device: Pytorch Device
    :param fold: Fold No. (for logging)
    :type fold: int
    :return: Loss after Epoch
    :rtype: float
    """
    # Put Model into eval mode
    model.eval()

    # Miscellaneous variables
    dataset_size = 0
    running_loss = 0.0

    # Iterate over dataloader
    for _, data in track(enumerate(dataloader), total=len(dataloader)):  # type: ignore
        # Move tensors to device
        ids = data["ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.long)

        # Get Batch Size
        batch_size = ids.size(0)

        with torch.cuda.amp.autocast():
            # Forward Pass
            outputs = model(ids, mask)

            # Calculate Loss
            loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        # Log the metrics
        wandb.log({f"Valid/Fold-{fold} Loss": epoch_loss})

    # Garbage Collection
    _ = gc.collect()

    return epoch_loss


def run_training(
    cfg: Dict,
    model: nn.Module,
    train_loader: Iterable,
    valid_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device,
    num_epochs: int,
    fold: int,
) -> Tuple[nn.Module, DefaultDict]:
    """
    Run the training pipeline

    :param cfg: Configuration Dictionary
    :type cfg: Dict
    :param model: PyTorch Model
    :type model: nn.Module
    :param train_loader: Training Dataloader
    :type train_loader: Iterable
    :param valid_loader: Validation Dataloader
    :type valid_loader: Iterable
    :param optimizer: PyTorch Optimizer
    :type optimizer: torch.optim.Optimizer
    :param scheduler: PyTorch Scheduler
    :param device: PyTorch Device
    :param num_epochs: Number of Epochs to train for
    :type num_epochs: int
    :param fold: Fold No. (for logging)
    :type fold: int
    :return: Model and a History Dictionary
    :rtype: Tuple[nn.Module, DefaultDict]
    """

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}\n")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(
            cfg,
            model,
            optimizer,
            scheduler,
            scaler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            fold=fold,
        )

        val_epoch_loss = valid_one_epoch(model, valid_loader, device=device, fold=fold)

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            wandb.run.summary["Best Loss"] = best_epoch_loss  # type: ignore
            best_model_wts = copy.deepcopy(model.state_dict())
            path = f"output/Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), path)
            # Save a model file from the current directory
            print("Model Saved")

        print()

    print(f"Best Loss: {best_epoch_loss}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
