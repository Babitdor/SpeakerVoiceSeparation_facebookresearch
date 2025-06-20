# Prepare Data

- In this project, WSJ0-orignal dataset was it.
  
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

# Bash scripts
- To essentially work and convert .wv1,.wv2 file formats of Wsj0-original Dataset to .wav format (which is the required format)
- Generate the mix, s1, s2, basically prepping the data in the above structure.

- **Convert WSJ0 .wv1/.wv2 to .wav:**
  ```bash
  bash dataset/wv1_to_wav.sh
  ```
- **Generate mixtures and individual speakers**
  ```bash
  bash dataset/prep_mix_spk.sh
