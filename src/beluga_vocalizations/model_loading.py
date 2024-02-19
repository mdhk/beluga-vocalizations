import json
import torch
from torch import nn
from torchaudio.models import wav2vec2_model


class AvesEmbedding(nn.Module):
    def __init__(self, config_path, model_path):
        super().__init__()
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(False)
        self.audio_sr = 16000
        self.frame_len = 0.02
        self.emb_dim = 768

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_features will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        return out