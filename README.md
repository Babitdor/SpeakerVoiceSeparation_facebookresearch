<<<<<<< HEAD
# TranSVoice: Speech Separation

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
=======
# Voice Separation with an Unknown Number of Multiple Speakers (ICML 2020)

We provide a [PyTorch][pytorch] implementation of the paper: [Voice Separation with an Unknown Number of Multiple Speakers][arxiv] In which, we present a new method for separating a mixed audio sequence, in which multiple voices speak simultaneously. The new method employs gated neural networks that are trained to separate the voices at multiple processing steps, while maintaining the speaker in each output channel fixed. A different model is trained for every number of possible speakers, and the model with the largest number of speakers is employed to select the actual number of speakers in a given sample. Our method greatly outperforms the current state of the art, which, as we show, is not competitive for more than two speakers. Please note that this implementation does not contain the "IDloss" as described in the paper.

Audio samples can be found here: [Samples][web]

<p align="center">
<img src="./img/arch.png" alt="The architecture of our network. The audio is being convolved with a stack of 1D convolutions and reordered by cutting
overlapping segments of length K in time, to obtain a 3D tensor. In our method, the RNN blocks are of the type of multiply and add.
After each pair of blocks, we apply a convolution D to the copy of the activations, and obtain output channels by reordering the chunks
and then using the overlap and add operator."
width="100%"></p>

## Installation

First, install Python 3.7 (recommended with Anaconda).

Clone this repository and install the dependencies. We recommend using
a fresh virtualenv or Conda environment.

```bash
git clone git@github.com:fairinternal/svoice.git
cd svoice
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt  
```

## Setup

### Configuration

We use [Hydra][hydra] to control all the training configurations. If you are not familiar with Hydra
we recommend visiting the Hydra [website][hydra-web].
Generally, Hydra is an open-source framework that simplifies the development of research applications
by providing the ability to create a hierarchical configuration dynamically.

The config file with all relevant arguments for training our model can be found under the `conf` folder.
Notice, under the `conf` folder, the `dset` folder contains the configuration files for
the different datasets. You should see a file named `config.yaml` with the relevant configuration for the debug sample set.

You can pass options through the
command line, for instance `python train.py lr=1e-4`.
Please refer to [conf/config.yaml](conf/config.yaml) for a reference of the possible options.
You can also directly edit the `config.yaml` file, although this is not recommended
due to the way experiments are automatically named, as explained hereafter.

### Checkpointing

Each experiment will get a unique name based on the command line options you passed.
Restarting the same command will reuse the existing folder and automatically
start from a previous checkpoint if possible. In order to ignore previous checkpoints,
you must pass the `restart=1` option.
Note that options like `device`, `num_workers`, etc. have no influence on the experiment name.

### Setting up a new dataset

If you want to train using a new dataset, you can:
1. Create a separate config file for it.
2. Place the new config files under the `dset` folder. Check [conf/dset/debug.yaml](conf/dset/debug.yaml)
for more details on configuring your dataset.
3. Point to it either in the general config file or via the command line, e.g. `./train.py dset=name_of_dset`.

You also need to generate the relevant `.json`files in the `egs/`folder.
For that purpose you can use the `python -m svoice.data.audio` command that will
scan the given folders and output the required metadata as json.
For instance, if your mixture files are located in `$mix` and the separated files are in `$spk1` and `$spk2`, you can do

```bash
out=egs/mydataset/tr
mkdir -p $out
python -m svoice.data.audio $mix > $out/mix.json
python -m svoice.data.audio $spk1 > $out/s1.json
python -m svoice.data.audio $spk1 > $out/s1.json
```

## Usage
### Quick Start with Toy Example
1. Run `./make_debug.sh` to generate json files for the toy dataset.
2. Run `python train.py`

Notice, we already provided the yaml file for it. Can be found under `conf/dset/debug.yaml`.

### Data Structure
The data loader reads both mixture and separated json files named: `mix.json` and `s<id>.json` where `<id>` is a running identifier. These files should contain all the paths to the wav files to be used to optimize and test the model along with their size (in frames).
You can use `python -m svoice.data.audio FOLDER_WITH_WAV1 [FOLDER_WITH_WAV2 ...] > OUTPUT.json` to generate those files.
You should generate the above files for both training and test sets (and validation set if provided). Once this is done, you should create a yaml (similarly to `conf/dset/debug.yaml`) with the dataset folders' updated paths.
Please check [conf/dset/debug.yaml](conf/dset/debug.yaml) for more details.

#### WSJ Mixture Generation 
In case you have access to the origin wsj0 data (sphere format), you can generate the mixtures using the tools provided in the following [repository][convtas] (see usage section in the readme). 
You can access the csv files containing all the metadata for generating the mixtures from the following [samples page][web].


### Training
Training is simply done by launching the `train.py` script:

```
python train.py
```

This will automaticlly read all the configurations from the `conf/config.yaml` file. You can override different configuration arguments from the command, this will automaticlly generate new folder using the override params.
```
python train.py lr=0.001
python train.py dset=librimix lr=0.001 swave.R=8
```

#### Distributed Training

To launch distributed training you should turn on the distributed training flag. This can be done as follows:

```
python train.py ddp=1
```

#### Logs

Logs are stored by default in the `outputs` folder. Look for the matching experiment name.
In the experiment folder you will find the training checkpoint `checkpoint.th` (containing the last state as well as the best state)
as well as the log with the metrics `trainer.log`. All metrics are also extracted to the `history.json`
file for easier parsing. Enhancements samples are stored in the `samples` folder (if `mix_dir` or `mix_json`
is set in the dataset config yaml file).

### Evaluating

Evaluating the models can be done by launching the following:

```
python -m svoice.evaluate <path to the model> <path to folder containing mix.json and all target separated channels json files s<ID>.json>
```

For more details regarding possible arguments, please see:

```
usage: Evaluate separation performance using MulCat blocks [-h]
                                                           [--device DEVICE]
                                                           [--sdr SDR]
                                                           [--sample_rate SAMPLE_RATE]
                                                           [--num_workers NUM_WORKERS]
                                                           [-v]
                                                           model_path data_dir

positional arguments:
  model_path            Path to model file created by training
  data_dir              directory including mix.json, s1.json, s2.json, ...
                        files

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --sdr SDR
  --sample_rate SAMPLE_RATE
                        Sample rate
  --num_workers NUM_WORKERS
  -v, --verbose         More loggging
```

### Separation

Separating files can be done by launching the following:

```
python -m svoice.separate --model_path=<path to the model> --mix_dir=<path to the dir with the mixture files> --out_dir=<path to store the separated files>
```

Notice, you can either provide `mix_dir` or `mix_json` for the test data.
For more details regarding possible arguments, please see:

```
usage: Speech separation using MulCat blocks [-h] --model_path MODEL_PATH
                                             [--mix_dir MIX_DIR]
                                             [--mix_json MIX_JSON]
                                             [--out_dir OUT_DIR]
                                             [--device DEVICE]
                                             [--sample_rate SAMPLE_RATE]
                                             [--batch_size BATCH_SIZE] [-v]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Model name
  --mix_dir MIX_DIR     Directory including mix wav files
  --mix_json MIX_JSON   Json file including mix wav files
  --out_dir OUT_DIR     Directory putting enhanced wav files
  --device DEVICE
  --sample_rate SAMPLE_RATE
                        Sample rate
  --batch_size BATCH_SIZE
                        Batch size
  -v, --verbose         More loggging
```


## Results 
Using the default configuration (same one as presented in our [paper][arxiv]), results should be similar to the following. 
All reprted numbers are the Scale-Invariant Signal-to-Noise-Ratio improvment (SI-SNRi) over the input mixture.

| Model       | #params | 2spk  | 3spk  | 4spk  | 5spk  |
|-------------|---------|-------| ------| ------| ------|
| ADANet      | 9.1M    | 10.5  | 9.1   |   -   |   -   |
| DPCL++         | 13.6M   | 10.8  | 7.1   |   -   |   -   |
| CBLDNN-GAT     | 39.5M   | 11.0  |   -   |   -   |   -   |
| TasNet         | 32.0M   | 11.2  |   -   |   -   |   -   |
| IBM            | -       | 13.0  | 12.8  | 10.6  | 10.3  |
| IRM            | -       | 12.7  | 12.5  |  9.8  |  9.6  |
| ConvTasNet     | 5.1M    | 15.3  | 12.7  |  8.5  |  6.8  |
| FurcaNeXt      | 51.4M   | 18.4  |   -   |   -   |   -   |
| DPRNN          | 3.6M    | 18.8  | 14.7  | 10.4  |  8.7  |
| <b>Ours        | <b> 7.5M    | <b>20.1  | <b>16.9  | <b>12.9  | <b>10.6  |


#### Learning Curves 
The following learning cures were obtained using L=8 (the encoder kernel size): 

<p align="center">
<img src="./img/lc.png" alt="Training curves of our model."
width="49%">
<img src="./img/sisnr.png" alt="SI-SNRi curves of our model."
width="49%">
</p>


## Citation
If you find our code useful for you research, please cite it as:

```
@inproceedings{nachmani2020voice,
  title={Voice Separation with an Unknown Number of Multiple Speakers},
  author={Nachmani, Eliya and Adi, Yossi and Wolf, Lior},
  booktitle={Proceedings of the 37th international conference on Machine learning},
  year={2020}
}
```

## License
This repository is released under the CC-BY-NC-SA 4.0. license as found in the [LICENSE](LICENSE) file.

The file: `svoice/models/sisnr_loss.py` and `svoice/data/preprocess.py` were adapted from the [kaituoxu/Conv-TasNet][convtas] repository. It is an unofficial implementation of the [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation][convtas-paper] paper, released under the MIT License.
Additionally, several input manipulation functions were borrowed and modified from the [yluo42/TAC][tac] repository, released under the CC BY-NC-SA 3.0 License.

[arxiv]: https://arxiv.org/abs/2003.01531
[web]: https://enk100.github.io/speaker_separation/
[pytorch]: https://pytorch.org/
[hydra]: https://github.com/facebookresearch/hydra
[hydra-web]: https://hydra.cc/
[convtas]: https://github.com/kaituoxu/Conv-TasNet 
[convtas-paper]: https://arxiv.org/pdf/1809.07454.pdf
[tac]: https://github.com/yluo42/TAC

>>>>>>> 920d15c (Initial commit)
