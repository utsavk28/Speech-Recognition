import torch
from torch import nn
from AudioHead import AudioHead
from TextHead import TextHead
from Classifier import Classifier
import numpy as np


class OverKill_Model(nn.Module):
    def __init__(self, audio_head_variant, text_head_variant, classifer_variant):
        super().__init__()
        self.audio_head = AudioHead(audio_head_variant)
        self.text_head = TextHead(text_head_variant)
        self.classifier = Classifier(classifer_variant)

    def forward(self, waveform, sampling_rate, text):
        audio_emb = self.audio_head.run(waveform, sampling_rate)
        text_emb = self.text_head.run(text)
        emb = torch.hstack((audio_emb, text_emb))
        op = self.classifier.forward(emb)
        return op

    def predict(self, waveform, sampling_rate, text):
        op = self.forward(waveform, sampling_rate, text)
        return op

    def print_trainable_params(self):
        def _printable_params(params):
            params = list(str(params))
            params.reverse()
            params = [''.join(params[i:i+3]) for i in range(0, len(params), 3)]
            params = ','.join(params)
            params = params[::-1]
            return params
        total_params, trainable_params = np.sum(np.array([
            self.audio_head.print_trainable_params(),
            self.text_head.print_trainable_params(),
            self.classifier.print_trainable_params()]), axis=0)
        print(f"Total Params: {_printable_params(total_params)}")
        print(f"Trainable Params: {_printable_params(trainable_params)}")
