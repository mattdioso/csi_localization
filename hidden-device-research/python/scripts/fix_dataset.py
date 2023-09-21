#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

combined_dir = "/home/matt/hidden-device-research/datasets/combined"
combined_fixed_dir = "/home/matt/hidden-device-research/datasets/combined_fixed"
file = combined_dir + "/bedroom.csv"
fixed_file = combined_fixed_dir + "/bedroom.csv"

#df = pd.read_csv(file, header=None)

#print(df[8].unique()[0])

#fixed_df = pd.read_csv(fixed_file, header=None)
#fixed_df.insert(8, column="8", value=df[8].unique()[0])
#print(fixed_df.head())
#fixed_df.to_csv(combined_fixed_dir + "/bedroom.csv", header=None, index=False)
#df.insert(8, column="8", value=0.0768)
#df.to_csv(combined_dir + "/bedroom.csv", header=None, index=False)

fixed_df = pd.read_csv(fixed_file, header=None)
print(fixed_df.head())
fixed_df[8] =1.5
print(fixed_df.head())
print(fixed_df[8].unique())
fixed_df.to_csv(fixed_file, header=None, index=False)
