# create a custom dataset

from torch.utils.data import Dataset  # for creating custom dataset
import pandas as pd  # for data processing
import torchaudio  # for audio processing
import os  # for file path


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):  # constructor
        self.annotations = pd.read_csv(annotations_file)  # load annotations file
        self.audio_dir = audio_dir  # set audio directory

    def __len__(self):  # return length of dataset
        return len(self.annotations)  # return number of samples in dataset

    def __getitem__(self, index):  # get a sample from the dataset
        audio_sample_path = self._get_audio_sample_path(index)  # get audio sample path
        label = self._get_audio_sample_label(index)  # get audio sample label
        signal, sr = torchaudio.load(audio_sample_path)  # load audio sample
        return signal, label  # return signal tensor and label

    def _get_audio_sample_path(self, index):  # get audio sample path
        fold = f"fold{self.annotations.iloc[index, 5]}"  # get fold number
        path = os.path.join(
            self.audio_dir, fold, self.annotations.iloc[index, 0]
        )  # create path using filename in annotations file
        return path  # return audio sample path

    def _get_audio_sample_label(self, index):  # get audio sample label
        return self.annotations.iloc[index, 6]  # return class label


if __name__ == "__main__":
    ANNOTAIONS_FILE = ""  # annotations file path
    AUDIO_DIR = ""  # audio directory path
    usd = UrbanSoundDataset(ANNOTAIONS_FILE, AUDIO_DIR)  # instantiate dataset
    print(f"There are {len(usd)} samples in the dataset.")  # print length of dataset
    signal, label = usd[0]  # get first sample

    a = 1
