#  Implement a CNN for Sound Classification
import torch
from torch import nn
from torchinfo import summary


class CNNNetwork(nn.Module):
    def __init__(self):  # constructor
        super().__init__()  # call parent constructor
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 1st convolutional layer
                in_channels=1,  # 1 channel for grayscale images
                out_channels=16,  # 16 filters in our convolutional layer
                kernel_size=3,  # kernel size of 3
                stride=1,  # stride of 1
                padding=2,  # padding of 2
            ),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2)  # max pooling layer
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 2nd convolutional layer
                in_channels=16,  # 1 channel for grayscale images
                out_channels=32,  # 16 filters in our convolutional layer
                kernel_size=3,  # kernel size of 3
                stride=1,  # stride of 1
                padding=2,  # padding of 2
            ),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2)  # max pooling layer
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(  # 3rd convolutional layer
                in_channels=32,  # 1 channel for grayscale images
                out_channels=64,  # 16 filters in our convolutional layer
                kernel_size=3,  # kernel size of 3
                stride=1,  # stride of 1
                padding=2,  # padding of 2
            ),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2), # max pooling layer
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(  # 4th convolutional layer
                in_channels=64,  # 1 channel for grayscale images
                out_channels=128,  # 16 filters in our convolutional layer
                kernel_size=3,  # kernel size of 3
                stride=1,  # stride of 1
                padding=2,  # padding of 2
            ),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2)  # max pooling layer
        )
        self.flatten = nn.Flatten()  # flatten the image tensors
        self.linear = nn.Linear(128 * 5 * 4, 10)  # linear layer 10classes
        self.softmax = nn.Softmax(dim=1)  # apply softmax function

    def forward(self, input_data):  # forward pass
        x = self.conv1(input_data)  # pass data to conv1
        x = self.conv2(x)  # pass data to conv2
        x = self.conv3(x)  # pass data to conv3
        x = self.conv4(x)  # pass data to conv4
        x = self.flatten(x)  # flatten the input data
        logits = self.linear(x)  # pass data to dense layers
        predictions = self.softmax(x)  # apply softmax function
        return predictions  # return predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get device
    cnn = CNNNetwork().to(device)  # instantiate model
    batch_size = 1 # batch size
    summary(cnn, input_size=(batch_size, 1, 64, 44), device=device) # print model summary
    print(f"Using Device: {device}") # print device
    print("------------------------------------------------------------------------------------")
    print(cnn)  # print model architecture