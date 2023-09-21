#!/usr/bin/env python3
import numpy as np
import pandas
from oct2py import octave

file = "1709_6.7_3.6_2.5_1_1_h_trace.mat"
octave.addpath('../../matlab/')
octave.addpath('../../datasets/mat_files/')
alldata=octave.extract_function(file)
