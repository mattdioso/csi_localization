#!/bin/bash

DIR="/Volumes/External Drive/research/mscse-capstone/hidden-device-research/results/tests/batch_321A/pcaps"
OUT_DIR="/Volumes/External Drive/research/mscse-capstone/hidden-device-research/results/tests/321A"

cd "$DIR"

for file in *; do
  tcpdump -r "$file" -c 4 -w "$OUT_DIR"/"$file"
done
