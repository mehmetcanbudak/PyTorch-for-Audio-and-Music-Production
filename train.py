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

BATCH_SIZE = 128  # number of data points in each batch


class FeedForwardNet(nn.Module):  # inherit from nn.Module
    def __init__(self):  # constructor
        super().__init__()  # call parent constructor
        self.flatten = nn.Flatten()  # flatten the image tensors
        self.dense_layers = nn.Sequential(  # create a sequence of layers
            nn.Linear(28 * 28, 256),  # 1st layer
            nn.ReLU(),  # activation function
            nn.Linear(256, 10),  # 2nd layer
        )
        self.softmax = nn.Softmax(dim=1)  # apply softmax function

    def forward(self, input_data):  # forward pass
        flattened_data = self.flatten(input_data)  # flatten the input data
        logits = self.dense_layers(flattened_data)  # pass data to dense layers
        predictions = self.softmax(logits)  # apply softmax function
        return predictions  # return predictions


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
    return train_data, validation_data  # return datasets


if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded successfully")

    # create data loader for the train set
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if GPU is available
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)  # create model and send it to device
