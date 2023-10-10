#!/usr/bin/env python3
import os
from collections import defaultdict
import pandas as pd
from oct2py import octave
import sys
import traceback
import numpy as np
mat_file_dir = "./mat_files"
csv_file_dir = "./csv_files"

octave.addpath('./matlab')
octave.addpath('./mat_files')
count = 0
for subdir, dirs, files in os.walk(mat_file_dir):
    parts = []
    for file in files:
        print(os.path.join(subdir, file))
        mat_file = os.path.join(subdir, file)
        try:
            alldata = octave.extract_function(mat_file)
        except Exception as e:
            alldata = ""

        if not isinstance(alldata, str):
            result_arr = np.empty([1, 1024])
            try:
                for i in range(0, 99):
                    try:
                        csi = alldata[0][i][0][0]['nss'][0]['data']
                    except Exception as e:
                        try:
                            print("failed nss 0, trying 3")
                            csi = alldata[0][i][0][0][3]['nss'][0]['data']
                        except:
                            try:
                                print("failed nss 3, trying 2")
                                csi = alldata[0][i][0][0][2]['nss'][0]['data']
                            except:
                                print("failed nss 2, trying 1")
                                csi = alldata[0][i][0][0][1]['nss'][0]['data']
                    arr_out = csi
                    result_arr = np.vstack((result_arr, arr_out))
            except:
                continue
            result_arr = np.delete(result_arr, 0, axis=0)
            print(result_arr.shape)
            csv_file_name = file[:-3] + "csv"
            pd.DataFrame(result_arr).to_csv(os.path.join(csv_file_dir, csv_file_name), index=False, header=None)
            count += 1
        else:
            print("alldata is empty")

print("processed: %d" % (count))
print("all done")


