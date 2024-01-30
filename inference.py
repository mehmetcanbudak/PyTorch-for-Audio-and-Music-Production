# Load back the previously created model and make inferences
import torch
import torchaudio

from cnn import CNNNetwork
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

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
    state_dict = torch.load(
        "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/cnnnet.pth",
        map_location=torch.device("cpu"),
    )  # load trained weights
    cnn.load_state_dict(state_dict)  # load weights into model

    # load UrbanSoundDataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu"
    )  # instantiate dataset

    # get a sample from the urban sound dataset for inference
    input, target = usd[0][0], usd[0][1]  # [batch_size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make inference
    predicted, expected = predict(cnn, input, target, class_mapping)  # make inference
    print(f"Predicted: {predicted}, Expected: {expected}")  # print prediction
