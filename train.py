import torch  # for deep learning
from torch import nn  # for neural network
from torch.utils.data import DataLoader  # for creating data loaders
from tqdm import tqdm  # for progress bar

BATCH_SIZE = 128  # number of data points in each batch
EPOCHS = 10  # number of times to pass through the whole dataset
# LEARNING_RATE = 1e-3  # learning rate
LEARNING_RATE = 0.001  # learning rate


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):  # train one epoch
    for inputs, targets in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # send data to device

        # calculate loss
        predictions = model(inputs)  # forward pass
        loss = loss_fn(predictions, targets)  # calculate loss

        # bacpropogate loss and update weights
        optimiser.zero_grad()  # reset gradients
        loss.backward()  # backpropogate loss
        optimiser.step()  # update weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):  # train model
    for i in range(epochs):  # train for number of epochs
        print(f"Epoch {i+1}")  # print epoch number
        train_one_epoch(
            model, data_loader, loss_fn, optimizer, device
        )  # train one epoch
        print("---------------------------")  # print seperator
    print("Training is done.")  # print finished message


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

    # instantiate loss function and optimiser
    loss_fn = nn.CrossEntropyLoss()  # instantiate loss function
    # optimiser = torch.optim.SGD(
    #     feed_forward_net.parameters(), lr=1e-3
    # )  # instantiate optimiser with SGD
    optimiser = torch.optim.Adam(
        feed_forward_net.parameters(), lr=LEARNING_RATE
    )  # instantiate optimiser with Adam

    # train model
    train(
        feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS
    )  # train model

    # save trained model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet_mnist.pth")  # save model
    print(
        "Model trained and saved successfully at feedforwardnet_mnist.pth"
    )  # print finished message
