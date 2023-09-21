#!/usr/bin/env python3
import os
import numpy
from oct2py import octave
from pprint import pprint

path = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/"
#file = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/mat_files/1709/1709_6.7_3.6_2.5_9_4_nlos_8_trace.mat"
octave.addpath("../../matlab/")
octave.addpath(path)
#octave.test_function(file)

redo_list = []

for file in os.listdir(path):
  filename = os.fsdecode(file)
  if filename.endswith(".mat"):
    print(os.path.join(path, filename))
    try:
      octave.test_function(os.path.join(path, filename))
    except:
      print("REDO: " + filename)
      redo_list.append(filename)

print("need to redo: " + str(len(redo_list)))
pprint(redo_list)
