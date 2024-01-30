# create a custom dataset
import torch
from torch.utils.data import ConcatDataset, Dataset  # for creating custom dataset
import pandas as pd  # for data processing
import torchaudio  # for audio processing
import os  # for file path

class UrbanSoundDataset(Dataset):
    def __init__(
        self, annotations_file, audio_dir, transformation, target_sample_rate
    ):  # constructor
        self.annotations = pd.read_csv(annotations_file)  # load annotations file
        self.audio_dir = audio_dir  # set audio directory
        self.transformation = transformation  # set transformation
        self.target_sample_rate = target_sample_rate  # set target sample rate

    def __len__(self):  # return length of dataset
        return len(self.annotations)  # return number of samples in dataset

    def __getitem__(self, index):  # get a sample from the dataset
        audio_sample_path = self._get_audio_sample_path(index)  # get audio sample path
        label = self._get_audio_sample_label(index)  # get audio sample label
        signal, sr = torchaudio.load(audio_sample_path)  # load audio sample
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)  # resample signal if necessary
        signal = self._mix_down_if_necessary(signal)  # mix down signal if necessary
        signal = self.transformation(signal)  # apply transformation
        return signal, label  # return signal and label

    def _resample_if_necessary(self, signal, sr):  # resample signal if necessary
        if sr != self.target_sample_rate:  # check sample rate
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate
            )  # instantiate resampler
            signal = resampler(signal)  # resample signal
        return signal

    def _mix_down_if_necessary(self, signal):  # mix down signal if necessary
        if signal.shape[0] > 1:  # check channels
            signal = torch.mean(signal, dim=0, keepdim=True)  # average the channels
        return signal

    def _get_audio_sample_path(self, index):  # get audio sample path
        fold = f"fold{self.annotations.iloc[index, 5]}"  # get fold number
        path = os.path.join(
            self.audio_dir, fold, self.annotations.iloc[index, 0] 
        )  # create path using filename in annotations file
        return path  # return audio sample path

    def _get_audio_sample_label(self, index):  # get audio sample label
        return self.annotations.iloc[index, 6]  # return class label


if __name__ == "__main__":
    ANNOTAIONS_FILE = "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K.csv"
    AUDIO_DIR = "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K/audio/"
    SAMPLE_RATE = 16000  # sample rate of audio file

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,  # sample rate of audio file
        n_fft=1024,  # number of samples in each fourier transform
        hop_length=512,  # number of samples to shift window by
        n_mels=64,  # number of mel spectrogram bands
    )
    # ms = mel_spectogram(signal)

    usd = UrbanSoundDataset(
        ANNOTAIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE
    )  # instantiate dataset
    print(f"There are {len(usd)} samples in the dataset.")  # print length of dataset
    signal, label = usd[0]  # get first sample
        
    #print(f"Sample Rate: {SAMPLE_RATE}") # print sample rate

    df = pd.read_csv("/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K.csv")
    print("Some samples from the dataset:")
    print(f"{df.head()}")