"""Dataset Definitions"""
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase

__all__ = ["FeedBackDataset", "prepare_loaders"]


class FeedBackDataset(torch.utils.data.Dataset):
    """Custom Torch Dataset Instance"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        """
        Utility Torch Dataset Class

        :param dataframe: pandas DataFrame
        :type dataframe: pd.DataFrame
        :param tokenizer: Tokenizer to use
        :type tokenizer: PreTrainedTokenizerBase
        :param max_length: Maximum Length for Tokenizer
        :type max_length: int
        """
        self.dataframe = dataframe
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = dataframe["discourse_text"].values
        self.essay = dataframe["essay_text"].values
        self.targets = dataframe["discourse_effectiveness"].values

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        discourse = self.discourse[index]
        essay = self.essay[index]
        # Add Seperation Token
        text = discourse + " " + self.tokenizer.sep_token + " " + essay

        # Encode Text
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
        )

        # Split inputs
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        target = self.targets[index]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }


def prepare_loaders(
    cfg: Dict, dataframe: pd.DataFrame, fold: int
) -> Tuple[Iterable, Iterable]:
    """prepare_loaders Utility Function to get dataloaders

    :param cfg: Configuration dictionary
    :type cfg: Dict
    :param dataframe: Dataframe with kfold entries
    :type dataframe: pd.DataFrame
    :param fold: K-Fold
    :type fold: int
    :return: Training and Valiation Dataloaders
    :rtype: Tuple[Iterable, Iterable]
    """

    # Get dataframes corresponding to the fold
    dataframe_train = dataframe[dataframe.kfold != fold].reset_index(drop=True)
    dataframe_valid = dataframe[dataframe.kfold == fold].reset_index(drop=True)

    # Create Torch Datasets
    train_dataset = FeedBackDataset(
        dataframe_train, tokenizer=cfg["tokenizer"], max_length=cfg["max_length"]
    )
    valid_dataset = FeedBackDataset(
        dataframe_valid, tokenizer=cfg["tokenizer"], max_length=cfg["max_length"]
    )

    # Create Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["train_batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg["valid_batch_size"],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader
