# beluga-vocalizations
Exploring Beluga whale vocal repertoires through machine listening

## Setup
To start using this repo:
1. Clone it locally:
   ```
   git clone git@github.com:mdhk/beluga-vocalizations.git
   ```
2. Download the audio and model files that are not stored in this repo into `data/` and `models/` directories, i.e. folder structure should look like this:
   ```
   .
    ├── data/
    │   └── Contact_calls_Mar16/
    ├── models/
    │   └── aves-base-bio/
    │       ├── models/aves-base-bio/aves-base-bio.torchaudio.model_config.json
    │       └── models/aves-base-bio/aves-base-bio.torchaudio.pt
    ├── notebooks/
    └── ... [etc.]
   ```
   The files in the `models/aves-base-bio/` directory are available from the [AVES model repository](https://github.com/earthspecies/aves) (TorchAudio versions).
3. Configure the virtual environment — with [Poetry installed](https://python-poetry.org/docs/#installing-with-the-official-installer), run the following commands in this repository's root directory:
    ```
    poetry config virtualenvs.in-project true
    poetry env use python3.10
    poetry install
    ```
    This will install dependencies and configure the environment as specified in [`pyproject.toml`](https://github.com/mdhk/beluga-vocalizations/blob/main/pyproject.toml).
4. Use the virtual environment — either

   a) before running any code, activate the environment in the Terminal with `source .venv/bin/activate`, then run scripts as usual; or

   b) use `poetry run`, e.g.
    - for scripts:
        ```
        poetry run python <path/to/script.py>
        ```
    -  for notebooks:
        ```
        poetry run jupyter lab
        ```
