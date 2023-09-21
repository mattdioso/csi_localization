#!/usr/bin/env python3
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from horizon_model import HorizonNet
import horizon_post_proc
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import butter, lfilter, freqz

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
  max_v = maximum_filter(signal, size=r, mode='wrap')
  pk_loc = np.where(max_v == signal)[0]
  pk_loc = pk_loc[signal[pk_loc] > min_v]
  if N is not None:
    order = np.argsort(-signal[pk_loc])
    pk_loc = pk_loc[order[:N]]
    pk_loc = pk_loc[np.argsort(pk_loc)]
  return pk_loc, signal[pk_loc]

def augment(x_img, flip, rotate):
  x_img = x_img.numpy()
  aug_type = ['']
  x_imgs_augmented = [x_img]
  if flip:
    aug_type.append('flip')
    x_imgs_augmented.append(np.flip(x_img, axis=-1))
  for shift_p in rotate:
    shift = int(round(shift_p * x_img.shape[-1]))
    aug_type.append('rotate %d' % shift)
    x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
  return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type

def augment_undo(x_imgs_augmented, aug_type):
  x_imgs_augmented = x_imgs_augmented.cpu().numpy()
  sz = x_imgs_augmented.shape[0] // len(aug_type)
  x_imgs = []
  for i, aug in enumerate(aug_type):
    x_img = x_imgs_augmented[i*sz : (i+1)*sz]
    if aug == 'flip':
      x_imgs.append(np.flip(x_img, axis=-1))
    elif aug.startswith('rotate'):
      shift = int(aug.split()[-1])
      x_imgs.append(np.roll(x_img, -shift, axis=-1))
    elif aug == '':
      x_imgs.append(x_img)
    else:
      raise NotImplementedError()
  return np.array(x_imgs)

def load_images(image_files):
  loaded_images = []
  for file in image_files:
    image = Image.open(file)
    max_size = (640, 480)
    image = image.resize(max_size)
    print(image.size)
    x = np.clip(np.asarray(image, dtype=float) / 255, 0, 1)
    loaded_images.append(x)
  return np.stack(loaded_images, axis=0)

def to_multichannel(i):
  if i.shape[2] == 3: return i
  i = i[:,:,0]
  return np.stack((i,i,i), axis=2)

def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
  import matplotlib.pyplot as plt
  import skimage
  from skimage.transform import resize

  plasma = plt.get_cmap('plasma')
  shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

  all_images = []

  for i in range(outputs.shape[0]):
    imgs = []
    if isinstance(inputs, (list, tuple, np.ndarray)):
      x = to_multichannel(inputs[i])
      x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
      imgs.append(x)

    if isinstance(gt, (list, tuple, np.ndarray)):
      x = to_multichannel(gt[i])
      x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
      imgs.append(x)

    if is_colormap:
      rescaled = outputs[i][:,:,0]
      if is_rescale:
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)
      imgs.append(plasma(rescaled)[:,:,:3])
    else:
      imgs.append(to_multichannel(outputs[i]))

    img_set = np.hstack(imgs)
    all_images.append(img_set)

  all_images = np.stack(all_images)
  return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def visualize_a_data(x, y_bon, y_cor):
  x = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
  y_bon= y_bon.numpy()
  y_bon = ((y_bon / np.pi + 0.5) * x.shape[0]).round().astype(int)
  y_cor = y_cor.numpy()
  gt_cor = np.zeros((30, 1024, 3), np.uint8)
  gt_cor[:] = y_cor[0][None, :, None] * 255
  img_pad = np.zeros((3, 1024, 3), np.uint8) + 255
  img_bon = (x.copy() * 0.5).astype(np.uint8)
  y1 = np.round(y_bon[0]).astype(int)
  y2 = np.round(y_bon[1]).astype(int)
  y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
  y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
  img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
  img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255

  return np.concatenate([gt_cor, img_pad, img_bon], 0)

def load_trained_model(Net, path):
  state_dict = torch.load(path, map_location='cpu')
  net = Net(**state_dict['kwargs'])
  net.load_state_dict(state_dict['state_dict'])
  return net

def find_vanishing_points(sorted_arr):
  np_arr = np.array(sorted_arr)
  np_arr -= math.ceil(np.average(np_arr))
  step = np.hstack((np.ones(len(np_arr)), -1 * np.ones(len(np_arr))))
  np_arr_step = np.convolve(np_arr, step, mode='valid')

  step_idx = np.argmax(np_arr_step)
  left_b = sorted_arr[:step_idx-1]
  right_b = sorted_arr[step_idx:]
  left_x = int(np.median(left_b))
  right_x = int(np.median(right_b))
  return left_x, right_x

def dense_model_scale_up(scale, images):
  from skimage.transform import resize
  scaled = []
  for i in range(len(images)):
    img = images[i]
    output_shape = (scale * img.shape[0], scale * img.shape[1])
    scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))
  return np.stack(scaled)

def DepthNorm(x, maxDepth):
  return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
  if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
  if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
  predictions = model.predict(images, batch_size=batch_size)
  return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def find_peaks(col_slice):
  np_arr = 10 * np.array(col_slice)

  arr_std = (np_arr - np_arr.mean()) / np_arr.std()
  arr_denoise = denoise_tv_chambolle(arr_std, weight=1)
  arr_step = -2 * np.cumsum(arr_denoise)
  step_indicator = arr_step == arr_step.max()
  n = np_arr.shape[0]
  plt.subplot(211)
  plt.plot(range(n-1), np.diff(np_arr), label='standardized')
#  print(np.diff(np_arr))
#  plt.plot(range(n), arr_denoise, label='denoised (TV)')
#  plt.legend()
  plt.grid()
  plt.subplot(212)
  plt.step(range(n), step_indicator)
  plt.show()

  '''
  np_arr -= math.ceil(np.average(np_arr))
  step = np.hstack((np.ones(len(np_arr)), -1 * np.ones(len(np_arr))))
  np_arr_step = np.convolve(np_arr, step, mode='valid')
  step_idx = np.argmax(np_arr_step)
  print(step_idx)
  min_idx = np.argmin(np_arr_step)
  print(min_idx)
  plt.plot(np_arr)
  plt.plot(np_arr_step/10)
  plt.plot((step_idx, step_idx), (np_arr_step[step_idx]/10, 0), 'r')
  plt.show()
  '''

def butter_lowpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y



def find_plateaus(arr, min_length=130, tolerance=0.98, smoothing=27):
  smooth_arr = uniform_filter1d(arr, size=smoothing)
  df = uniform_filter1d(np.gradient(smooth_arr), size=smoothing)
  d2f = uniform_filter1d(np.gradient(df), size=smoothing)
  def zero_runs(x):
    iszero = np.concatenate(([0], np.equal(x, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

  eps = np.quantile(abs(d2f), tolerance)
  smalld2f = (abs(d2f) <= eps)
  p = zero_runs(np.diff(smalld2f))

  plt.plot(range(len(arr)), arr, color='red')
  plt.plot(range(len(smooth_arr)), smooth_arr, color='blue')
#  plt.plot(range(len(d2f)), d2f, color='green')
  plt.xlabel("pixel along depth image y-axis")
  plt.ylabel("normalized depth value. 0.1 = 1m")
  plt.show()

  plateaus = p[(np.diff(p) > min_length).flatten()]
  return plateaus

def find_larger_plateau(plat_arr):
  if len(plat_arr) == 1: return plat_arr[0]
  ret = plat_arr[0]
  for plat in plat_arr:
    if abs(plat[1] - plat[0]) > abs(ret[1] - ret[0]):
      ret = plat

  return ret

def calc_real_world_coor(u, v, z):
  #Principal point information for camera intrinsics on a Galaxy S8
  C = [314.11, 231.10]
  #focal length information from camera intrinsics on a Galaxy S8
  f = [494.19, 494.41]
  # z = value from Depth map, Z is scaled value for real-world distance
  Z = z * 10
  #Need to rotate on z-axis to align coordinate systems, all it requires is a negation on x-axis
  u  = -u
  X = ((u - C[0]) * Z)/f[0]
  Y = ((v - C[1]) * Z)/f[1]

  return [X, Y, Z]

def calc_3d_distance(p1, p2):
  return np.sqrt(sum(math.pow(a-b, 2) for a, b in zip (p1, p2)))

#previously, i was calculating length simply taking the depth captured at the very center pixel (straight ahead from the camera)
#what if there's something between the back wall and the camera (i.e. a bookshelf)?
# this new method instead takes a few knowns, half the calculated width of the room and the depth point along the seam of the walls
# for use in pythagorean calculation of the length.
def calc_length(half_width, corner_depth):
  print(type(half_width))
  print(type(corner_depth))
  return np.sqrt(math.pow(corner_depth, 2) - math.pow(half_width, 2))

def get_camera_intrinsics():
  C = [314.11, 231.10]
  f = [494.19, 494.41]
  return C, f
