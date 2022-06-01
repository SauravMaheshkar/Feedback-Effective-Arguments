"""Training Code"""
import gc
from typing import Dict

import torch
from rich import print
from torch.optim import AdamW
from transformers import AutoTokenizer

import wandb
from src.io.data import get_dataframe
from src.io.dataset import prepare_loaders
from src.nn.engine import run_training
from src.nn.model import FeedBackModel
from src.nn.schedulers import fetch_scheduler
from src.utils import set_seed

# Configuration
config: Dict = {
    "seed": 42,
    "epochs": 5,
    "model_name": "roberta-base",
    "strategy": "GroupKFold",
    "train_batch_size": 8,
    "valid_batch_size": 16,
    "max_length": 512,
    "learning_rate": 1e-5,
    "scheduler": "CosineAnnealingWarmRestarts",
    "min_lr": 1e-6,
    "T_max": 500,
    "T_0": 25,
    "weight_decay": 1e-6,
    "n_fold": 4,
    "n_accumulate": 1,
    "num_classes": 3,
}

# Tokenizer
config["tokenizer"] = AutoTokenizer.from_pretrained(config["model_name"])


if __name__ == "__main__":
    # Miscellaneous
    set_seed(seed=config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize a Weights & Biases Run
    wandb.init(
        project="FeedBack-Effective-Arguments",
        job_type="train",
        group=config["model_name"],
        tags=[config["model_name"], config["strategy"]],
        config=config,
    )

    # Get Dataframe
    df = get_dataframe(
        strategy=config["strategy"], num_splits=config["n_fold"], seed=config["seed"]
    )

    # Training Pipeline
    for fold in range(config["n_fold"]):

        # Create Dataloaders
        train_loader, valid_loader = prepare_loaders(
            cfg=config, dataframe=df, fold=fold
        )

        # Instantiate Model
        model = FeedBackModel(config)
        model.to(device)

        # Get Number of Parameters
        trainable_model_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Model Parameters:{trainable_model_params}")
        wandb.run.summary["Model Parameters"] = trainable_model_params  # type: ignore

        # Define Optimizer and Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = fetch_scheduler(
            scheduler=config["scheduler"],
            optimizer=optimizer,
            min_lr=config["min_lr"],
            t_max=config["T_max"],
            t_0=config["T_0"],
        )

        # Training !!!
        model, history = run_training(
            config,
            model,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            device=device,
            num_epochs=config["epochs"],
            fold=fold,
        )

        del model, history, train_loader, valid_loader
        _ = gc.collect()

    # Log Average Validation Loss
    VALID_LOSS = 0.0

    for fold in range(config["n_fold"]):
        VALID_LOSS += wandb.run.summary[f"Valid/Fold-{fold} Loss"]  # type: ignore

    VALID_LOSS /= config["n_fold"]
    wandb.run.summary["Validation Loss"] = VALID_LOSS  # type: ignore
    print(f"Average Validation Loss: {VALID_LOSS}")

    wandb.finish()
