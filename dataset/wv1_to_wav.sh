#! /bin/bash
root_dir="csr_1_LDC93S6A/csr_1"
out_dir="wav_data/csr_1_LDC93S6A/csr_1"
sph2pipe="tools/sph2pipe/sph2pipe.exe"

mkdir -p "$out_dir"

for disc in "$root_dir"/*; do
    disc_name=$(basename "$disc")
    if [[ "$disc_name" == "readme.txt" || "$disc_name" == "11-13.1" || "$disc_name" == "file.tbl" ]]; then
        continue
    fi
    wsj0_dir="$disc/wsj0"
    if [[ ! -d "$wsj0_dir" ]]; then
        continue
    fi

    for subdir in "$wsj0_dir"/*; do
        subdir_name=$(basename "$subdir")
        if [[ ! "$subdir_name" =~ ^si && ! "$subdir_name" =~ ^sd ]]; then
            continue
        fi
        for spk in "$subdir"/*; do
            [ -d "$spk" ] || continue
            echo "Processing speaker directory: $spk"
            for wv_file in "$spk"/*.wv1 "$spk"/*.wv2; do
                [ -f "$wv_file" ] || continue
                echo "Checking file: $wv_file"
                base=$(basename "$wv_file")
                ext="${base##*.}"
                stem="${base%.*}"
                if [[ "$ext" == "wv1" ]]; then
                    target_name="${stem}.wav"
                else
                    target_name="${stem}_1.wav"
                fi
                # Compute relative path from $root_dir and build output path
                rel_path="${wv_file#$root_dir/}"
                rel_dir=$(dirname "$rel_path")
                out_spk_dir="$out_dir/$rel_dir"
                mkdir -p "$out_spk_dir"
                target_path="$out_spk_dir/$target_name"
                # Preview print
                echo "Converting: $wv_file -> $target_path"
                "$sph2pipe" -f wav "$wv_file" "$target_path"
            done
        done
    done
done