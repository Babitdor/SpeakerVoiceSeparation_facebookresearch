*Prepare Data**

- **Convert WSJ0 .wv1/.wv2 to .wav:**
  ```bash
  bash dataset/wv1_to_wav.sh
  ```
- **Generate mixtures:**
  ```bash
  bash dataset/prep_mix_spk.sh
  # or
  python tools/pywsj0_mix/generate_wsjmix.py -p csr_1_LDC93S6A/csr_1 -o mix_splits_data -n 2 -sr 8000 --len_mode min max
