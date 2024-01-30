# Load back the previously created model and make inferences
import torch

from train_mnist import FeedForwardNet, download_mnist_datasets

# class_mapping = ["rock", "classical", "jazz", "pop"] # class mapping
class_mapping = ["0", "1", "2", "3", "4", "5" "6", "7", "8", "9"]  # class mapping


def predict(model, input, target, class_mapping):  # make prediction
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # don't calculate gradients
        predictions = model(input)  # forward pass
        # Tensor object with specific dimension. 2D Tensor. 1st dimension is number of samples, 2nd dimension is number of classes
        # Tensor (1, 10) -> ([0.1], 0.01, ......, 0.6]]
        predicted_index = predictions[0].argmax()  # get index with max value
        predicted = class_mapping[predicted_index]  # get predicted class
        expected = class_mapping[target]  # get expected class
        return predicted, expected  # return prediction and expected


if __name__ == "__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()  # instantiate model
    state_dict = torch.load("feedforwardnet_mnist.pth")  # load trained weights
    feed_forward_net.load_state_dict(state_dict)  # load weights into model

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()  # download validation dataset

    # get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]  # get first sample
    # sample = next(iter(validation_data))  # get a sample from the dataset

    # make inference
    predicted, expected = predict(
        feed_forward_net, input, target, class_mapping
    )  # make inference
    print(f"Predicted: {predicted}, Expected: {expected}")  # print prediction
