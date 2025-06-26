# SVoice: Speech Separation

## Overview

**SVoice** is a deep learning-based speech separation toolkit built with PyTorch and Hydra. It is designed for training, evaluating, and experimenting with source separation models (such as Conv-TasNet and Demucs variants) on datasets like WSJ0 and custom mixtures.
This version includes a novel enhancement: the integration of Transformers within the Dual-Path architecture, replacing traditional Bi-LSTM blocks to better capture long-range temporal dependencies in speech.

---

## Features

- **Flexible configuration** using [Hydra](https://hydra.cc/)
- **Support for multiple models** (e.g., Conv-TasNet, Demucs, SVoice custom models)
- **Transformer-based Dual-Path model support
- **Training, validation, and evaluation** pipelines
- **SI-SNR, PESQ, STOI metrics**
- **Automatic checkpointing and logging**
- **Data preprocessing scripts for WSJ0 and custom datasets**
- **Bash and Python utilities for data preparation and conversion**

---

## Project Structure

```
svoice/
├── conf/                # Hydra configuration files (config.yaml, etc.)
├── dataset/             # Data preparation scripts and raw data folders
├── egs/                 # Example recipes and experiment scripts
├── transvoice/              # Core source code (models, solver, evaluation, utils)
├── scripts/             # Additional scripts (e.g., make_dataset.py)
├── tools/               # External tools (e.g., sph2pipe)
├── outputs/             # Hydra output directory for logs, checkpoints, samples
├── train.py             # Main training script
├── check.py             # Utility to check JSON data sizes
└── README.md            # Project documentation
```

---

## Dataset Structure

The project expects datasets to be organized in a **2-speaker mixture format** (or more, if configured).  
A typical directory structure for 2-speaker data is:

```
dataset/
└── 2speakers/
    └── wav8k/
        └── min/ or max/
            ├── tr/   # Training set
            │   ├── mix/   # Mixture wav files
            │   ├── s1/    # Speaker 1 source wav files
            │   └── s2/    # Speaker 2 source wav files
            ├── cv/   # Validation set
            │   ├── mix/
            │   ├── s1/
            │   └── s2/
            └── tt/   # Test set
                ├── mix/
                ├── s1/
                └── s2/
```

- **mix/** contains the mixed audio files.
- **s1/** and **s2/** contain the clean source files for each speaker.
- This structure is compatible with WSJ0-2mix and similar datasets.

---

## Getting Started

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
# For colored logging
pip install colorlog
```

### 2. **Prepare Data**

- **Convert WSJ0 .wv1/.wv2 to .wav:**
  ```bash
  bash dataset/wv1_to_wav.sh
  ```
- **Generate mixtures:**
  ```bash
  bash dataset/prep_mix_spk.sh
  # or
  python tools/pywsj0_mix/generate_wsjmix.py -p csr_1_LDC93S6A/csr_1 -o mix_splits_data -n 2 -sr 8000 --len_mode min max
  ```

### 3. **Configure Your Experiment**

Edit `conf/config.yaml` to set:
- Model type and parameters
- Dataset paths (see above for 2-speaker structure)
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- Logging and checkpointing options

### 4. **Train a Model**

```bash
python train.py
```

Hydra will create a new output directory for each run under `outputs/`.

### 5. **Check Logs and Outputs**

- Training logs: `outputs/exp_*/trainer.log`
- Checkpoints: `outputs/exp_*/checkpoint.th`
- Samples: `outputs/exp_*/samples/`

### 6. **Evaluate or Analyze Data**

- Use `check.py` to inspect JSON data sizes:
  ```bash
  python egs/mydataset/check.py
  ```

---

## Data Generation Script

The `scripts/make_dataset.py` script can be used to generate synthetic mixtures with room simulation and noise:

**Example usage:**
```bash
python scripts/make_dataset.py \
  --in_path dataset/2speakers/wav8k/min/tr/s1 \
  --out_path dataset/tmp \
  --noise_path dataset/wham_noise/tr \
  --num_of_speakers 2 \
  --num_of_scenes 10 \
  --sec 4 \
  --sr 8000
```
- `--in_path` should point to the directory containing clean speaker wav files (e.g., `s1` or `s2`).
- `--out_path` is where the generated mixtures and sources will be saved.
- `--noise_path` should point to a directory with noise wav files.
- The script will create mixtures and sources in the same 2-speaker directory structure as above.

---

## Configuration

All experiment settings are controlled via `conf/config.yaml` using Hydra.  
Key sections include:

- **Dataset and preprocessing**
- **Model and optimizer settings**
- **Logging and checkpointing**
- **Evaluation metrics**

See comments in `config.yaml` for details.

---

## Utilities

- **wv1_to_wav.sh**: Converts WSJ0 .wv1/.wv2 files to .wav, preserving directory structure.
- **prep_mix_spk.sh**: Prepares mixture and speaker folders for training.
- **check.py**: Checks the size of JSON data splits.
- **make_dataset.py**: Generates synthetic mixtures with room simulation and noise.

---

## Tips

- To change the output directory, edit the `hydra.run.dir` field in `config.yaml`.
- To enable PESQ computation, set `pesq: true` in your config.
- For colored logs, ensure `colorlog` is installed.

---

## Citation

If you use this codebase in your research, please cite the original SVoice paper and any relevant upstream projects (e.g., Conv-TasNet, Demucs, Hydra).

```bibtex
@inproceedings{nachmani2020voice,
  title={Voice Separation with an Unknown Number of Multiple Speakers},
  author={Nachmani, Eliya and Adi, Yossi and Wolf, Lior},
  booktitle={Proceedings of the 37th international conference on Machine learning},
  year={2020}
}
```
---

## License

This repository is released under the CC-BY-NC-SA 4.0. license as found in the [LICENSE](LICENSE) file.

The file: `svoice/models/sisnr_loss.py` and `svoice/data/preprocess.py` were adapted from the [kaituoxu/Conv-TasNet][convtas] repository. It is an unofficial implementation of the [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation][convtas-paper] paper, released under the MIT License.
Additionally, several input manipulation functions were borrowed and modified from the [yluo42/TAC][tac] repository, released under the CC BY-NC-SA 3.0 License.

---

## Acknowledgements

- [Hydra](https://hydra.cc/)
- [WSJ0 dataset](https://catalog.ldc.upenn.edu/LDC93S6A)
- [Conv-TasNet](https://github.com/funcwj/Conv-TasNet)
- [Demucs](https://github.com/facebookresearch/demucs)
- [colorlog](https://github.com/borntyping/python-colorlog)
