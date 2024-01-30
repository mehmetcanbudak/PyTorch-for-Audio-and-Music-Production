# Load back the previously created model and make inferences
import torch

from cnn import CNNNetwork
from train import _

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]  # class mapping


def predict(model, input, target, class_mapping):  # make prediction
    model.eval()  # set model to evaluation mode
    with torch.no_grad():  # don't calculate gradients
        predictions = model(input)  # forward pass
        predicted_index = predictions[0].argmax()  # get index with max value
        predicted = class_mapping[predicted_index]  # get predicted class
        expected = class_mapping[target]  # get expected class
    return predicted, expected  # return prediction and expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
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
