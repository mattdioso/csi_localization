#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import open3d as o3d
from yolo_detect import detect
import utils
from keras.models import load_model
from layers import BilinearUpSampling2D
from scipy.spatial.distance import pdist
import math
import itertools

def gen_depth_map(depth_model, img_fp):
  orig_img = utils.load_images([img_fp])
  output = utils.predict(depth_model, orig_img)
  output = utils.dense_model_scale_up(2, output)
  depth_img = output[0, :, :, 0]
  return depth_img

def calc_obj_volume(obj, depth_map):
  pcd = []
  obj_cor = obj['obj_cor']
  C, f = utils.get_camera_intrinsics()
  z_max_val = 0
  y_max_val = 0
  V = 0
  for i in range(int(float(obj_cor["y1"])), int(float(obj_cor["y2"]))):
    for j in range(int(float(obj_cor["x1"])), int(float(obj_cor["x2"]))):
      z = depth_map[i][j]
      x = (j - C[0]) * z / f[0]
      y = (i - C[1]) * z / f[1]
      pcd.append([x, y, z])
      if z > z_max_val:
        z_max_val = z
      if y > y_max_val:
        y_max_val = y

  pcd_o3d = o3d.geometry.PointCloud()
  pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

  z_filter_num = z_max_val * 0.8
  y_filter_num = y_max_val * 0.8

  bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [0, z_filter_num]]
  bounding_box_points = list(itertools.product(*bounds))
  bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bounding_box_points))

  pcd_cropped = pcd_o3d.crop(bounding_box)
  bb = pcd_cropped.get_axis_aligned_bounding_box()
  bb.color=[1,0,0]
  pts = pcd_cropped.get_axis_aligned_bounding_box().get_box_points()
#  print(np.asarray(pts))i
  if np.argmax(np.asarray(pts)) == 0:
    bb = pcd_o3d.get_axis_aligned_bounding_box()
    bb.color=[1,0,0]
    pts = bb.get_box_points()
#  if obj['obj_name'] == 'tv':
#    o3d.visualization.draw_geometries([pcd_o3d, bb])

  pts_dist = np.unique(pdist(pts))
 #   print(pts_dist)
  try:
    H = pts_dist[0] * 10
    W = pts_dist[1] * 10
    L = pts_dist[2] * 10

    print("WIDTH: " + str(W))
    print("HEIGHT: " + str(H))
    print("LENGTH: " + str(L))

    print(obj['obj_name'] + "\tVOLUME: " + str(W*H*L))
    V = W * H * L
  except:
    print(obj['obj_name'] + " size is negligent")
    print(obj_cor)

  return V

if __name__ == '__main__':
  imgs = ['../input/test_2/front.png', '../input/test_2/back.png']
  custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
  nyu_dense_model = load_model('../model/nyu.h5', custom_objects=custom_objects, compile = False)
  total_v = 0
  for img in imgs:
    depth_img = gen_depth_map(nyu_dense_model, img)
    detected_obj = detect(source=img)
    objects = detected_obj["objects"]
    for obj in objects:
      print(obj['obj_name'])
      total_v += calc_obj_volume(obj, depth_img)

  print("Total volume of all detected objects: " + str(total_v) + " m^3")
