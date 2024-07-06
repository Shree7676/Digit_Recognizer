import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values
    X_test = test_df.values

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Set any value greater than 0 to 1
    X_train[X_train > 0] = 1
    X_test[X_test > 0] = 1

    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, y_train, X_test


class MNISTDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.long) if labels is not None else None
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.images[idx], self.labels[idx]
        return self.images[idx]


def get_data_loaders(X_train, y_train, X_test, batch_size=128, val_split=0.1):
    train_dataset = MNISTDataset(X_train, y_train)
    test_dataset = MNISTDataset(X_test)

    train_data, val_data = train_test_split(
        range(len(train_dataset)), test_size=val_split, random_state=42
    )
    train_sampler = torch.utils.data.SubsetRandomSampler(train_data)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
