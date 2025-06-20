#! /bin/bash

src_path="csr_1_LDC93S6A/csr_1"
out_path="wav_files"

mkdir -p "$out_path"

for subdir in "$src_path"/*; do
    name=$(basename "$subdir")
    if [[ "$name" == "readme.txt" || "$name" == "11-13.1" || "$name" == "file.tbl" ]]; then
        continue
    fi
        python tools/pywsj0_mix/generate_wsjmix.py -p "$src_path"/"$name" -o "$out_path" -n 2 -sr 8000 --len_mode "min" "max"
done