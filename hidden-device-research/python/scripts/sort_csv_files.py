#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from collections import defaultdict

csv_file_dir = "/home/matt/hidden-device-research/datasets/csv_fixed"
combined_file_dir = "/home/matt/hidden-device-research/datasets/combined_fixed"

envs = defaultdict(list)

for subdir, dirs, files, in os.walk(csv_file_dir):
  for file in files:
    #print(file)
    env_id, L, W, H, grid, antenna, los, num_trace, end = file.split("_")
    envs[env_id].append(file)

for key, vals in envs.items():
  data = []
  for val in vals:
    df_in = pd.read_csv(os.path.join(csv_file_dir, val), header=None)
    for index, row in df_in.iterrows():
      data.append(row)
  pd.DataFrame(data).to_csv(os.path.join(combined_file_dir, key + ".csv"), index=False, header=None)
