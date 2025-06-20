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

- **Convert WSJ0 .wv1/.wv2 to .wav:**
  ```bash
  bash dataset/wv1_to_wav.sh
  ```
- **Generate mixtures:**
  ```bash
  bash dataset/prep_mix_spk.sh
  # or
  python tools/pywsj0_mix/generate_wsjmix.py -p csr_1_LDC93S6A/csr_1 -o mix_splits_data -n 2 -sr 8000 --len_mode min max
