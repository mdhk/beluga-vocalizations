from beluga_vocalizations.paths import (
    ROOT, 
    BELUGA_AUDIO
)

import glob
import numpy as np
import torchaudio
from pathlib import Path
import pandas as pd

filenames = [Path(fp).name for fp in sorted(glob.glob(str(BELUGA_AUDIO / '*.wav')))]
call_classes = np.empty(len(filenames), dtype=object)
call_types = np.empty(len(filenames), dtype=object)
sample_rates = np.empty(len(filenames), dtype=int)
durations = np.empty(len(filenames), dtype=np.float64)
for i, fn in enumerate(filenames):
    call_classes[i], call_types[i] = fn.split('.')[3].split('-')
    y, sample_rates[i] = torchaudio.load(BELUGA_AUDIO / fn)
    durations[i] = y.shape[-1] / sample_rates[i]
recordings_df = pd.DataFrame.from_dict({
    'filename': filenames,
    'call_class': call_classes,
    'call_type': call_types,
    'sample_rate': sample_rates,
    'duration': durations
})
recordings_df.index.name = 'voc_idx'
recordings_df.to_csv(ROOT / 'results/recordings.csv')