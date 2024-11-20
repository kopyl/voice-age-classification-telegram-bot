# source: https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    y = y.detach().cpu().numpy()
    return y


def convert_pydub_into_numpy_signal(pydub_audio, sampling_rate):
    pydub_audio = pydub_audio.set_frame_rate(sampling_rate)
    numpy_signal = np.array(pydub_audio.get_array_of_samples())
    if pydub_audio.channels == 2:
        numpy_signal = numpy_signal.reshape((-1, 2)).mean(axis=1)
    return pydub_audio.frame_rate, np.float32(numpy_signal) / 2**15


def predict_age_from_pydub_audio(pydub_audio) -> int:
    sampling_rate, numpy_signal = convert_pydub_into_numpy_signal(pydub_audio, sampling_rate=16000)
    predictions = process_func(numpy_signal, sampling_rate)
    age = int(predictions[0][0].item() * 100)
    return age


device = 'cuda'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name)
model.to(device)