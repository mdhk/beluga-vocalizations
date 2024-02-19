from beluga_vocalizations.paths import (
    ROOT, 
    BELUGA_AUDIO,
    AVES_BIO_CONFIG, 
    AVES_BIO_MODEL
)
from beluga_vocalizations.data_loading import load_recordings_df, get_dataloader
from beluga_vocalizations.model_loading import AvesEmbedding

import torch
import numpy as np
import pandas as pd

rec_df = load_recordings_df(ROOT / 'results/recordings.csv').reset_index()

embedding_model = AvesEmbedding(config_path=AVES_BIO_CONFIG,
                                model_path=AVES_BIO_MODEL)
embedding_model.eval()

if torch.cuda.is_available():
    embedding_model.cuda()

dl = get_dataloader(rec_df, embedding_model.audio_sr, ["voc_idx", "call_class", "call_type"])

voc_idx = []
emb_idx = []
embeddings = []
for rec in dl:
    x = rec['x'].cuda()
    voc = rec['voc_idx'].item()
    with torch.no_grad():
        out = embedding_model(x).cpu().detach().numpy()
    N_frames = out.shape[1]
    start_times = np.arange(0, embedding_model.frame_len*(N_frames+1), embedding_model.frame_len)[:N_frames]
    voc_idx.extend([voc]*N_frames)
    emb_idx.extend([str(voc).zfill(4) + '_' + "{:.2f}".format(st) for st in start_times])
    embeddings.extend([out[:, i, :].squeeze() for i in range(N_frames)])

emb_df = pd.DataFrame(np.vstack(embeddings), 
                      columns=[f'emb{d+1}' for d in range(embedding_model.emb_dim)])
emb_df.insert(0, 'emb_idx', emb_idx)
emb_df.insert(1, 'voc_idx', voc_idx)
emb_df = emb_df.set_index('emb_idx')

emb_df.to_csv(ROOT / 'results/AVES_frame_embeddings.csv')