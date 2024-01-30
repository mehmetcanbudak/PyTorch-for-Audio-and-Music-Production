# 1- download dataset
# 2- create data loader model
# 3- build model
# 4- train model
# 5- save trained model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",  # path to dataset
        download=True,  # download if not exist
        train=True,  # train dataset
        transform=ToTensor(),  # convert data to tensor
    )
    validation_data = datasets.MNIST(
        root="data",  # path to dataset
        download=True,  # download if not exist
        train=False,  # don't train dataset
        transform=ToTensor(),  # convert data to tensor
    )
    return train_data, validation_data


if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded successfully")

    # create data loader for the train set
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
