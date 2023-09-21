#!/usr/bin/env python3
import os
from collections import defaultdict
import pandas as pd
from oct2py import octave
import sys
import traceback
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
mat_file_dir = "/home/matt/hidden-device-research/datasets/mat_files/bedroom"
csv_file_dir = "/home/matt/hidden-device-research/datasets/csv_fixed"

#file format is env identifier_L_W_H_grid_antenna_hidden or visible_num_trace.pcap
# ['608', '7.3', '3.9', '3.0', '10', '1', 'los', '10', 'trace.mat']
octave.addpath('../../matlab/')
octave.addpath('../../datasets/mat_files/')
count = 0
for subdir, dirs, files in os.walk(mat_file_dir):
  parts = []
  for file in files:
    #parts = file.split("_")
#    print(file)
    env_id, L, W, H, grid, antenna, los, num_trace, end = file.split("_")
    """
    env_id = parts[0]
    L = parts[1]
    W = parts[2]
    H = parts[3]
    grid = parts[4]
    antenna = parts[5]
    los = parts[6]
    num_trace = parts[7]
    """
    print(os.path.join(subdir, file))
    mat_file = os.path.join(subdir, file)
    info_arr = np.array([env_id, L, W, H, grid, antenna, los, num_trace])
    try:
      test_file = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/mat_files/1709/1709_6.7_3.6_2.5_10_2_los_10_trace.mat"
      alldata = octave.extract_function(mat_file)
      # [0][i for num packets][0][0]['nss'][0]
    except Exception as e:
      alldata=""

    if not isinstance(alldata, str):
      result_arr = np.empty([1, 1032])
      #print(info_arr)
      try:
        for i in range(0, 99):
          #print(alldata[0][0][0][0][3]['nss'][0]['data'].shape)
          try:
            csi = alldata[0][i][0][0]['nss'][0]['data']
          except Exception as e:
            #print(e)
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
          #print(csi)
          arr_out = np.concatenate((info_arr, csi), axis=None)
          result_arr = np.vstack((result_arr, arr_out))
        #print(arr_out)
      except:
        continue
      result_arr = np.delete(result_arr, 0, axis=0)
      print(result_arr.shape)
      csv_file_name = file[:-3] + "csv"
      #print(csv_file_name)
      pd.DataFrame(result_arr).to_csv(os.path.join(csv_file_dir, csv_file_name), index=False, header=None)
      count += 1
    else:
      print("alldata is empty")
      #except Exception as e:
      #print(e)
    #print(alldata)
print("processed: %d" % (count))
print("all done")
