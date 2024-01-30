# create a custom dataset
import os  # for file path

import pandas as pd  # for data processing
import torch
import torchaudio  # for audio processing
from torch.utils.data import ConcatDataset  # for creating custom dataset
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        target_sample_rate,
        num_samples,
        device,
    ):  # constructor
        self.annotations = pd.read_csv(annotations_file)  # load annotations file
        self.audio_dir = audio_dir  # set audio directory
        self.device = device  # set device
        self.transformation = transformation.to(
            self.device
        )  # set transformation to device
        self.target_sample_rate = target_sample_rate  # set target sample rate
        self.num_samples = num_samples  # set number of samples

    def __len__(self):  # return length of dataset
        return len(self.annotations)  # return number of samples in dataset

    def __getitem__(self, index):  # get a sample from the dataset
        audio_sample_path = self._get_audio_sample_path(index)  # get audio sample path
        label = self._get_audio_sample_label(index)  # get audio sample label
        signal, sr = torchaudio.load(audio_sample_path)  # load audio sample
        signal = signal.to(self.device)  # send signal to device
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000)
        signal = self._resample_if_necessary(signal, sr)  # resample signal if necessary
        signal = self._mix_down_if_necessary(signal)  # mix down signal if necessary
        signal = self._cut_if_necessary(signal)  # cut signal if necessary
        signal = self._right_pad_if_necessary(signal)  # right pad signal if necessary
        signal = self.transformation(signal)  # apply transformation
        return signal, label  # return signal and label

    def _cut_if_necessary(self, signal):  # cut signal if necessary
        # signal -> Tensor -> (1, num_samles) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:  # check signal length
            signal = signal[:, : self.num_samples]  # cut signal
        return signal

    def _right_pad_if_necessary(self, signal):  # right pad signal if necessary
        length_signal = signal.shape[1]  # get signal length
        if length_signal < self.num_samples:  # check signal length
            num_missing_samples = (
                self.num_samples - length_signal
            )  # calculate number of missing samples
            last_dim_padding = (0, num_missing_samples)  # create padding
            signal = torch.nn.functional.pad(
                signal, last_dim_padding
            )  # pad signal on last dimension
        return signal

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
    ANNOTAIONS_FILE = "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K/UrbanSound8K.csv"
    AUDIO_DIR = (
        "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K/audio/"
    )
    # SAMPLE_RATE = 16000 # sample rate of audio file
    SAMPLE_RATE = 22050  # sample rate of audio file
    NUM_SAMPLES = 22050  # number of samples

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,  # sample rate of audio file
        n_fft=1024,  # number of samples in each fourier transform
        hop_length=512,  # number of samples to shift window by
        n_mels=64,  # number of mel spectrogram bands
    )
    # ms = mel_spectogram(signal)

    usd = UrbanSoundDataset(
        ANNOTAIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )  # instantiate dataset
    print(f"There are {len(usd)} samples in the dataset.")  # print length of dataset
    signal, label = usd[0]  # get first sample

    # print(f"Sample Rate: {SAMPLE_RATE}") # print sample rate

    df = pd.read_csv(
        "/Users/mehmetcanbudak/Projects/Mehmetcan/PyTorch_Audio/UrbanSound8K/UrbanSound8K.csv"
    )
    print("Some samples from the dataset:")
    print(f"{df.head()}")
