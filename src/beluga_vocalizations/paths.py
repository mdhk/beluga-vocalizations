from beluga_vocalizations.utils import get_project_root

ROOT = get_project_root()
BELUGA_AUDIO = ROOT / 'data/Contact_calls_Mar16'
AVES_BIO_CONFIG = ROOT / 'models/aves-base-bio/aves-base-bio.torchaudio.model_config.json'
AVES_BIO_MODEL = ROOT / 'models/aves-base-bio/aves-base-bio.torchaudio.pt'