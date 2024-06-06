import torch
from torch import nn
from type import AudioHeadVariant
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperForCausalLM
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class AudioHead:
    def __init__(self, variant=None):
        self.processor = None
        self.model = None
        self.variant = variant
        if variant is AudioHeadVariant.WAV2VEC2:
            self._load_wav2vec_model()
        elif variant is AudioHeadVariant.WAV2VEC2_BERT:
            self._load_wav2vec_bert_model()
        elif variant is AudioHeadVariant.WHISPER:
            self._load_whisper_model()
        else:
            raise Exception("Invalid AudioHeadVariant Type")
        self._freeze_model()

    def _load_wav2vec_model(self):
        model_name = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def _load_wav2vec_bert_model(self):
        pass

    def _load_whisper_model(self):
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-base")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base")
        self.model.config.forced_decoder_ids = None

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _aggregate(self, input):
        return torch.mean(input, dim=1)

    def run(self, waveforms, sampling_rate):
        t = None
        for i, sr in enumerate(sampling_rate):
            curr = self._run(waveforms[i], sr)
            if t is None:
                t = curr
            else:
                t = torch.cat((t, curr), dim=0)
        return t

    def _run(self, waveform, sampling_rate, return_tensors="pt"):
        input_features = self.processor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features)
        outputs = self.model(input_features, output_hidden_states=True,
                             decoder_input_ids=predicted_ids)
        return self._aggregate(outputs.encoder_last_hidden_state)

    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"{'='*50} {self.variant} {'='*50}")
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        return total_params, trainable_params
