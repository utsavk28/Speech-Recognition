import os
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.idx_to_text = {
            "01": "Kids are talking by the door",
            "02": "Dogs are sitting by the door"
        }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        audio_file_name = audio_path.split('\\')[-1]
        idx = audio_file_name.split('-')[4]
        return self.idx_to_text[idx]


class AudioDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.num_samples = 96000

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000
        waveform = waveform[:1].reshape(-1)
        waveform = self._right_pad(waveform)
        return waveform, sample_rate

    def _right_pad(self, waveform):
        signal_length = waveform.shape[0]
        if signal_length < self.num_samples:
            num_padding = self.num_samples-signal_length
            last_dim_padding = (0, num_padding)
            waveform = nn.functional.pad(waveform, last_dim_padding)
        return waveform


class SERDataset(Dataset):
    def __init__(self, directory):
        self.file_list = self._get_all_files(directory)
        self.audio_dataset = AudioDataset(self.file_list)
        self.text_dataset = TextDataset(self.file_list)
        self.id_to_idx = {
            '01': 0,
            '02': 1,
            '03': 2,
            '04': 3,
            '05': 4,
            '06': 5,
            '07': 6,
            '08': 7,
        }
        self.id_to_emotion = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                              '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

    def _get_all_files(self, directory):
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        audio_file_name = audio_path.split('\\')[-1]
        emotion_id = audio_file_name.split('-')[2]
        return *self.audio_dataset.__getitem__(idx), self.text_dataset.__getitem__(idx), self.id_to_idx[emotion_id]
