from beluga_vocalizations.paths import BELUGA_AUDIO

from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd

def load_recordings_df(rec_csv_file, audio_dir=BELUGA_AUDIO):
    rec_df = pd.read_csv(rec_csv_file, index_col='voc_idx')
    rec_df.insert(1, "filepath", (BELUGA_AUDIO / rec_df['filename'].astype(str)).astype(str))
    return rec_df

class Vox(Dataset):
    def __init__(self, dataset_dataframe, audio_sr, annotation_names):
        """ 
        Dataset for vocalization classification with AVES
        
        Source:
        https://colab.research.google.com/drive/1dtBorrZkEfsn90Mj9SETF2DFAY9sjCqe?usp=sharing

        Input
        -----
        dataset_dataframe (pandas dataframe): indicating the filepath, annotations and partition of a signal
        audio_sr (int): sampling rate expected by network
        annotation_name (list[str]): string corresponding to the annotation columns in the dataframe, e.g. ["call_type","recording_date"]
        """
        super().__init__()
        self.audio_sr = audio_sr
        self.annotation_names = annotation_names
        self.dataset_info = dataset_dataframe

    def __len__(self):
        return len(self.dataset_info)

    def get_one_item(self, idx):
      """ Load base audio """
      row = self.dataset_info.iloc[idx]
      x, sr = torchaudio.load(row["filepath"])
      if len(x.size()) == 2:
          x = x[0, :]
      if sr != self.audio_sr:
          x = torchaudio.functional.resample(x, sr, self.audio_sr)
      return x, row

    def __getitem__(self, idx):
        x, row = self.get_one_item(idx)
        out = {"x" : x, "filepath" : row['filepath'], "filename" : row['filename']}
        for k in self.annotation_names:
          out[k] = row[k]
        return out

def get_dataloader(dataset_dataframe, audio_sr, annotation_names):
    """
    Source: https://colab.research.google.com/drive/1dtBorrZkEfsn90Mj9SETF2DFAY9sjCqe?usp=sharing
    """
    return DataLoader(
            Vox(dataset_dataframe, audio_sr, annotation_names),
            batch_size=1,
            shuffle=False,
            drop_last=False
        )