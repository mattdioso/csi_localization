#!/usr/bin/env python3
import os
import numpy as np
from os.path import exists
import random
import math
import pickle
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

#dataset_dir = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/combined"
dataset_dir = "/home/matt/hidden-device-research/datasets/combined_fixed"
files = ["1709_mags.npy", "321A_mags.npy", "bedroom_mags.npy", "320_mags.npy", "608_mags.npy",  "321A_mags_augmented.npy", "bedroom_mags_augmented.npy","608_mags_augmented.npy", "1709_mags_augmented.npy"]
val_file = "320_mags_augmented.npy"

norm_files = ["1709_mags_norm.npy", "321A_mags_norm.npy", "bedroom_mags_norm.npy", "320_mags_norm.npy", "608_mags_norm.npy",  "321A_mags_norm_augmented.npy", "bedroom_mags_norm_augmented.npy","608_mags_norm_augmented.npy", "1709_mags_norm_augmented.npy"]
norm_val_file = "320_mags_norm_augmented.npy"


def clean_image(csi):
  result_arr = np.empty([1, 998])
  for i in range(csi.shape[0]):
    np_arr = csi[i][csi[i] != 0]
    try:
      result_arr = np.vstack((result_arr, np_arr))
    except:
      np_arr_shape = np_arr.shape[0]
      diff = 998 - np_arr_shape
      for i in range(diff):
        np_arr = np.append(np_arr ,0)
      result_arr = np.vstack((result_arr, np_arr))
  result_arr = np.delete(result_arr, 0, axis=0)
  return result_arr

def load_unseen_data(norm=True):
  X = []
  Y = []
  grids = { 1: 0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0 }
  count = 0
  if norm:
    validation_file = norm_val_file
  else:
    validation_file = val_file
  with open(os.path.join(dataset_dir, validation_file), 'rb') as f:
    np_arr = np.load(f, allow_pickle=True)
  for j in np_arr:
    cleaned_csi = clean_image(j)
    label = cleaned_csi[:, -1].ravel()
    features = cleaned_csi[:, :-1]
    actual_label = label.max()
    X.append(features)
    Y.append(actual_label)
    grids[actual_label] += 1
      #count += 1
    #else:
    #  break

  print(grids)

  X = np.array(X)
  Y = np.array(Y)
  print(np.unique(Y))
  return X, Y


def load_data():
  X_train = []
  Y_train = []
  x_test = []
  y_test = []
  all_sets = []
  all_labels = []
  for file in files:
    with open(os.path.join(dataset_dir, file), 'rb') as f:
      np_arr = np.load(f, allow_pickle=True)
    for j in np_arr:
      cleaned_csi = clean_image(j)
      all_sets.append(cleaned_csi)

  random.shuffle(all_sets)
  print(len(all_sets))
  #print(all_sets[0].shape)
  total = len(all_sets)
  train_size = math.floor(total * 0.4)
  test_size = math.floor(total*0.6)
  train_set = all_sets[:train_size]
  test_set = all_sets[train_size+1:]

  for sample in train_set:
    label = sample[:, -1].ravel()
    features = sample[:, :-1]
    actual_label = label.max()
    X_train.append(features)
    Y_train.append(actual_label)

  for sample in test_set:
    label = sample[:, -1].ravel()
    features = sample[:, :-1]
    actual_label = label.max()
    x_test.append(features)
    y_test.append(actual_label)

  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)

  print("X_train: " + str(len(X_train)))
  print("Y_train: " + str(len(Y_train)))
  print("x_test: " + str(len(x_test)))
  print("y_test: " + str(len(y_test)))

  print("-----SHAPES-----")
  print("X_train")
  print(X_train.shape)
  print("Y_train")
  print(Y_train.shape)
  print("X_test")
  print(x_test.shape)
  print("Y_test")
  print(y_test.shape)

  print(type(X_train))
  print(type(Y_train))
  print(np.unique(Y_train))
  return X_train, Y_train, x_test, y_test

def load_equal_data(norm=True):
  X_train = []
  Y_train = []
  x_test = []
  y_test= []
  all_sets = []
  all_features = []
  all_labels= []
  grids = { 1: 0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0 }

  if norm:
    files_arr = norm_files
  else:
    files_arr = files

  for file in files_arr:
    with open(os.path.join(dataset_dir, file), 'rb') as f:
      np_arr = np.load(f, allow_pickle=True)
    for j in np_arr:
      cleaned_csi = clean_image(j)
      all_sets.append(cleaned_csi)

  random.shuffle(all_sets)
  total = len(all_sets)
  print(total)
  train_size = math.floor(total * 0.6)
  test_size = math.floor(total * 0.4)

  for sample in all_sets:
    features = sample[:, :-1]
    labels = sample[:, -1]
    if len(np.unique(labels.ravel())) != 1:
      labels = np.max(labels)
      #sample[:, -1] = label
      #features = sample[:, :-1]
    else:
      labels = np.unique(labels.ravel())[0]
    all_features.append(features)
    all_labels.append(labels)

  print(math.ceil(train_size/21))
  grid_split = math.ceil(train_size/21)
  for f, l in zip(all_features, all_labels):
    l = int(l)
    if grids[l] != grid_split:
      X_train.append(f)
      Y_train.append(l)
      grids[l] = grids[l] + 1
    else:
      x_test.append(f)
      y_test.append(l)
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  x_test = np.array(x_test)
  y_test = np.array(y_test)

  print(grids)

  print("X_train: " + str(len(X_train)))
  print("Y_train: " + str(len(Y_train)))
  print("x_test: " + str(len(x_test)))
  print("y_test: " + str(len(y_test)))

  print("-----SHAPES-----")
  print("X_train")
  print(X_train.shape)
  print("Y_train")
  print(Y_train.shape)
  print("X_test")
  print(x_test.shape)
  print("Y_test")
  print(y_test.shape)

  #print(X_train[0])

  print(type(X_train))
  print(type(Y_train))
  print(np.unique(Y_train))
  return X_train, Y_train, x_test, y_test

  

if __name__ == '__main__':
  load_equal_data(norm=False) 
  #load_unseen_data()
