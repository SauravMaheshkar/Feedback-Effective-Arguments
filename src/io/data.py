"""Data Processing Functions"""
import os

import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

TRAIN_DIR: str = "data/competition/train"

__all__ = ["get_dataframe"]


def get_essay(essay_id: str) -> str:
    """
    Utility Function that fetches essay text corresponding to the ID

    :param essay_id: Essay ID from the dataframe
    :type essay_id: str
    :return: Essay Text from the .txt files
    :rtype: str
    """
    # get path of .txt file corresponding to the ID
    essay_path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    # read and return the text from the file
    essay_text = open(essay_path, "r").read()  # pylint: disable=R1732, W1514
    return essay_text


def get_dataframe(
    strategy: str = "StratifiedGroupKFold", num_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """
    Pre-process dataset with the provided Cross Validation Strategy

    :param strategy: Which Cross Validation strategy to use,
                    defaults to "StratifiedGroupKFold"
    :type strategy: str, optional
    :param num_splits: Number of folds to split, defaults to 5
    :type num_splits: int, optional
    :param seed: Random Seed for the process, defaults to 42
    :type seed: int, optional
    :raises ValueError: raises an error if an unimplemented strategy is provided
    :return: DataFrame split by folds
    :rtype: pd.DataFrame
    """

    # Read CSV
    dataframe = pd.read_csv("data/competition/train.csv")

    # Add Text from files
    dataframe["essay_text"] = dataframe["essay_id"].apply(get_essay)

    ### Perform Cross Validation ###

    # GroupKFold Strategy
    if strategy == "GroupKFold":
        gkf = GroupKFold(n_splits=num_splits)

        for fold, (_, val_) in enumerate(
            gkf.split(X=dataframe, groups=dataframe.essay_id)
        ):
            dataframe.loc[val_, "kfold"] = int(fold)

        dataframe["kfold"] = dataframe["kfold"].astype(int)

    # StratifiedGroupKFold Strategy
    elif strategy == "StratifiedGroupKFold":
        sgkf = StratifiedGroupKFold(
            n_splits=num_splits, shuffle=True, random_state=seed
        )

        for fold, (_, val_) in enumerate(
            sgkf.split(X=dataframe, groups=dataframe.essay_id)
        ):
            dataframe.loc[val_, "kfold"] = int(fold)

        dataframe["kfold"] = dataframe["kfold"].astype(int)

    # StratifiedKFold Strategy
    elif strategy == "StratifiedKFold":
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

        for fold, (_, val_) in enumerate(
            skf.split(X=dataframe, groups=dataframe.essay_id)  # pylint: disable=E1120
        ):
            dataframe.loc[val_, "kfold"] = int(fold)

        dataframe["kfold"] = dataframe["kfold"].astype(int)

    # ValueError
    else:
        raise ValueError(
            f"""{strategy}, is not a recognized strategy,
            please use StratifiedGroupKFold, GroupKFold or StratifiedKFold"""
        )

    # One-Hot Encode the labels
    encoder = LabelEncoder()
    dataframe["discourse_effectiveness"] = encoder.fit_transform(
        dataframe["discourse_effectiveness"]
    )

    # Serialize the Encoder
    with open("output/le.pkl", "wb") as filepath:
        joblib.dump(encoder, filepath)

    return dataframe
