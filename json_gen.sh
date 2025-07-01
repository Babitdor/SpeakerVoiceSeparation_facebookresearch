#!/bin/bash

for set in cv tr tt; do
    out=egs/mydataset/$set
    mkdir -p $out
    mix=dataset/2speakers/wav8k/max/$set/mix
    spk1=dataset/2speakers/wav8k/max/$set/s1
    spk2=dataset/2speakers/wav8k/max/$set/s2
    python -m svoice.data.audio $mix > $out/mix.json
    python -m svoice.data.audio $spk1 > $out/s1.json
    python -m svoice.data.audio $spk2 > $out/s2.json
done
