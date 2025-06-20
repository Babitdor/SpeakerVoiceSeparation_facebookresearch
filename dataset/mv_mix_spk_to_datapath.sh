#!/bin/bash
for split in max min; do
    for type in cv tr tt; do
        for src in mix s1 s2; do
            dst_dir="2speakers/wav8k/$split/$type/$src/"
            mkdir -p "$dst_dir"
            src_dir="mix_splits_data/*/2speakers/wav8k/$split/$type/$src/*.wav"
            cp $src_dir "$dst_dir"
        done
    done
done