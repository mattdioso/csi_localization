#!/usr/bin/env python3
import numpy as np
import os
import json
import pandas as pd
import matplotlib
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
import argparse
import glob
import utils
import horizon_post_proc
from horizon_model import HorizonNet
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
import sys
import cv2
import matplotlib.pyplot as plt
import scipy

THRESHOLD = 30
def is_within_range(number, max_val, threshold):
  percentage = ((max_val - number) * 100) / number
  return percentage <= threshold

def inference(net, x, device, flip=False, rotate=[], visualize=False,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
  H, W = tuple(x.shape[2:])
  x, aug_type = utils.augment(x, flip, rotate)
  y_bon_, y_cor_ = net(x.to(device))
  y_bon_ = utils.augment_undo(y_bon_.cpu(), aug_type).mean(0)
  y_cor_ = utils.augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)
  if visualize:
    vis_out = utils.visualize_a_data(x[0], torch.FloatTensor(y_bon_[0]), torch.FloatTensor(y_cor_[0]))
  else:
    vis_out = None

  y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
  y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
  y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
  y_cor_ = y_cor_[0, 0]

  z0 = 50
  _, z1 = horizon_post_proc.np_refine_by_fix_z(*y_bon_, z0)

  if force_raw:
    cor = np.stack([np.arange(1024), y_bon_[0]], 1)
  else:
    if min_v is None:
      min_v = 0 if force_cuboid else 0.05

    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = utils.find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    #Generate wall-walls
    cor, xy_cor = horizon_post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
      xy2d = np.zeros((len(xy_cor), 2), np.float32)
      for i in range(len(xy_cor)):
        xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
        xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
      if not Polygon(xy2d).is_valid:
        print('Fail to generate valid general layout!! '
              'Generate cuboid as fallback.',
              file=sys.stderr)
        xs_ = utils.find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
        cor, xy_cor = horizon_post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

  cor = np.hstack([cor, horizon_post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
  cor_id = np.zeros((len(cor) * 2, 2), np.float32)
  for j in range(len(cor)):
    cor_id[j*2] = cor[j, 0], cor[j, 1]
    cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
  cor_id[:, 0] /= W
  cor_id[:, 1] /= H

  return cor_id, z0, z1, vis_out

parser = argparse.ArgumentParser(description='preprocessing step for csi_localization')
parser.add_argument('--depth_model', default='../model/nyu.h5', type=str, help='Pre-trained model file from DenseDepth')
parser.add_argument('--layout_model', default='../model/resnet50_rnn__panos2d3d.pth', type=str, help='Pre-trained model file from HorizonNet')
parser.add_argument('--input', default='../input/input_2/*.png', type=str, help='Directory to test files')
parser.add_argument('--output_dir', default='../outputs/input_2', type=str, help='Directory to place produced images')
args = parser.parse_args()

print(args.depth_model)
print(args.layout_model)
print(args.input)

device = torch.device('cpu')
net = utils.load_trained_model(HorizonNet, args.layout_model).to(device)
net.eval()

paths = sorted(glob.glob(args.input))
if len(paths) == 0:
  print('no images found')
for path in paths:
  assert os.path.isfile(path), '%s not found' % path

if not os.path.isdir(args.output_dir):
  print('output directory %s not existed. Create one' %args.output_dir)
  os.makedirs(args.output_dir)

rgb_raw_imgs = []

with torch.no_grad():
  for i_path in tqdm(paths, desc='Inferencing'):
    k = os.path.split(i_path)[-1][:-4]

    img_pil = Image.open(i_path)
    if img_pil.size != (1024, 512):
      img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
    img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
    x = torch.FloatTensor([img_ori / 255])
    cor_id, z0, z1, vis_out = inference(net=net, x=x, device=device,
                                        flip=False, rotate=[],
                                        visualize=True, force_cuboid=False,
                                        force_raw=False,
                                        min_v=None, r=0.05)

    with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
      json.dump({
        'z0': float(z0),
        'z1': float(z1),
        'uv': [[float(u), float(v)] for u, v in cor_id],
      }, f)

    if vis_out is not None:
      vis_path = os.path.join(args.output_dir, k + '.raw.png')
      rgb_raw_imgs.append(vis_path)
      vh, vw = vis_out.shape[:2]
      Image.fromarray(vis_out)\
          .resize((vw//2, vh//2), Image.LANCZOS)\
          .save(vis_path)

W = 0
H = 0
L = 0

for img in rgb_raw_imgs:
  image = cv2.imread(img)
  image = cv2.resize(image, (640, 510))
  gt_image = image[:15, :]
  rows, cols, _ = gt_image.shape

  lower_b = np.array([220, 220, 220])
  upper_b = np.array([255, 255, 255])

  mask = cv2.inRange(gt_image, lower_b, upper_b)
  #cv2.imshow('mask', mask)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  idx = np.nonzero(mask)
  idx = np.transpose(idx)
#  print(np.unique(idx[0,:]))
  sorted_arr = np.sort(idx[:, 1])
  #x_cors for left and right corners
  print(sorted_arr)
  left, right = utils.find_vanishing_points(sorted_arr)
  height_col = left + 30
  '''
  image = image[31:, :]

  green_lower = np.array([0, 10, 0])
  green_upper = np.array([0, 255, 0])

  green_mask = cv2.inRange(image, green_lower, green_upper)
  print(np.nonzero(green_mask))
  cv2.imshow('image', green_mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  '''
  custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
  nyu_dense_model = load_model(args.depth_model, custom_objects=custom_objects, compile=False)
  print('\nModel loaded({0})'.format(args.depth_model))

  orig_img = "../input/608/3_4/" + os.path.split(img)[-1][:-8] + '.png'
  orig_imgs = utils.load_images([orig_img])
  output = utils.predict(nyu_dense_model, orig_imgs)
  output = utils.dense_model_scale_up(2, output)
  print(os.path.split(img)[-1][:-8])
  #np.savetxt(r'../outputs/' + os.path.split(img)[-1][:-8] + '_data.csv', output[0,:,:,0], delimiter=',')
  viz = utils.display_images(output.copy())
  matplotlib.image.imsave(os.path.split(img)[-1][:-8] + '_depth.png', viz)
  depth_img = output[0,:,:,0]
  left_arr = depth_img[:,left+50]
  right_arr = depth_img[:,right - 20]
  print(left)
  left_plateaus = utils.find_plateaus(left_arr)
  right_plateaus = utils.find_plateaus(right_arr)
  
  lp = utils.find_larger_plateau(left_plateaus)
  rp = utils.find_larger_plateau(right_plateaus)
  print("PLATEAUS ON THE LEFT: ")
  print(lp)
  print("PLATEAUS ON THE RIGHT: ")
  print(rp)
  height_cor = lp
  if abs(lp[1] - lp[0]) < abs(rp[1] - rp[0]):
    print("using right for height")
    height_col = right - 20
    height_cor = rp


  ## calculating width of room
  u1 = left
  v1 = height_cor[0]
  z1 = depth_img[v1, u1]
  p1 = utils.calc_real_world_coor(u1, v1, z1)
  u2 = right
  v2 = height_cor[0]
  z2 = depth_img[v2, u2]
  p2 = utils.calc_real_world_coor(u2, v2, z2)

  w = utils.calc_3d_distance(p1, p2)
  if w > W:
    W = w

  ## calculating height of room
  u1 = height_col
  v1 = height_cor[0]
  z1 = np.transpose(depth_img)[u1, v1]
  p1 = utils.calc_real_world_coor(u1, v1, z1)

  u2 = height_col
  v2 = height_cor[1]
  z2 = np.transpose(depth_img)[u2, v2]
  p2 = utils.calc_real_world_coor(u2, v2, z2)
#  print("(%0.2f, %0.2f, %0.2f)"% (u1, v1, z1))
#  print("(%0.2f, %0.2f, %0.2f)"% (u2, v2, z2))
#  print(p1)
#  print(p2)
  h = utils.calc_3d_distance(p1, p2)
  print("Height: " + str(h))
  print(p1)
  print(p2)
  if h > H:
    H = h

  ##calculate length of room (add midpoints of depth_img together)
  #L += (depth_img[320, 240] * 10)
  print(W/2)
  print(height_cor)
  L += utils.calc_length(W/2, depth_img[height_cor[0], 240] * 10)

  plt.imshow(depth_img)
  plt.show()
print("WIDTH: " + str(W))
print("HEIGHT: " + str(H))
print("LENGTH: " + str(L))

print("VOLUME: " + str(W*H*L))
