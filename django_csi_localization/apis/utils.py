import numpy as np
import os
import json
import pandas as pd
from PIL import Image
from scipy.ndimage.filters import maximum_filter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib
import apis.horizonnet_utils as horizonnet_utils
from shapely.geometry import Polygon
import cv2
import sys

import tensorflow as tf
from tensorflow.keras.models import load_model
from config.dense_layers import BilinearUpSampling2D
import torch
from config.horizon_model import HorizonNet
from config.yolo_model import attempt_load
from config.settings import load_horizon_net_model

import open3d as o3d


#from config.settings import model_objs

model_objs = dict()

dense_depth_custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

model_objs = {
    'dense_depth': load_model('./models/nyu.h5', custom_objects=dense_depth_custom_objects, compile=False),
    'horizon_net': load_horizon_net_model(HorizonNet, './models/resnet50_rnn__panos2d3d.pth').to(torch.device('cpu')),
    'yolov7': attempt_load('./models/yolov7.pt', map_location='cpu')
}

def is_within_range(number, max_val, threshold=30):
    percentage = ((max_val - number) * 100) / number
    return percentage <= threshold



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
        x_img = x_imgs_augmented[i*sz : (i+1) * sz]
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
        x = np.clip(np.asarray(image, dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i, i, i), axis=2)

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
            imgs.append(plasma(rescaled)[:,:,3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    return skimage.util.montage(all_images, fill=(0,0,0))

def visualize_a_data(x, y_bon, y_cor):
    x = (x.numpy().transpose([1,2,0]) * 255).astype(np.uint8)
    y_bon = y_bon.numpy()
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

def find_vanishing_points(sorted_arr):
    np_arr = np.array(sorted_arr)
    np_arr -= math.ceil(np.average(np_arr))
    step = np.hstack((np.ones(len(np_arr)), -1 * np.ones(len(np_arr))))
    np_arr_step = np.convolve(np_arr, step, mode='valid')

    step_idx = np.argmax(np_arr_step)
    left_b = sorted_arr[:step_idx - 1]
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
    step_indicator = arr_step == arr_step.all
    n = np_arr.shape[0]
    plt.subplot(211)
    plt.plot(range(n-1), np.diff(np_arr), label='standardized')
    plt.grid()
    plt.subplot(212)
    plt.step(range(n), step_indicator)
    plt.show()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_plateaus(arr, min_length=100, tolerance=0.95, smoothing=27):
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
  plt.savefig("plateau.png")
  #plt.show()

  plateaus = p[(np.diff(p) > min_length).flatten()]
  return plateaus

def find_larger_plateau(plat_arr):
    if len(plat_arr) == 1: return plat_arr[0]
    ret = plat_arr[0]
    for plat in plat_arr:
        if abs(plat[1] - plat[0]) > abs(ret[1] - ret[0]):
            ret = plat
    return ret

def get_camera_intrinsics():
    C = [314.11, 231.10]
    f = [494.19, 494.41]
    return C, f

def calc_real_world_coor(u, v, z):
    C, f = get_camera_intrinsics()
    Z = z * 10

    u = -u
    X = ((u - C[0]) * Z)/f[0]
    Y = ((v - C[1]) * Z)/f[1]

    return [X, Y, Z]

def calc_3d_distance(p1, p2):
    return np.sqrt(sum(math.pow(a-b, 2) for a, b in zip (p1, p2)))

def calc_length(half_width, corner_depth):
    return np.sqrt(math.pow(corner_depth, 2) - math.pow(half_width, 2))

def inference(net, x, device, flip=False, rotate=[], visualize=False, force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    H, W = tuple(x.shape[2:])
    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)
    if visualize:
        vis_out = visualize_a_data(x[0], torch.FloatTensor(y_bon_[0]), torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None
    
    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]

    z0 = 50
    _, z1 = horizonnet_utils.np_refine_by_fix_z(*y_bon_, z0)

    if force_raw:
        cor = np.stack([np.arange(1024), y_bon_[0]], 1)
    else:
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05
        
        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        cor, xy_cor = horizonnet_utils.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
        if not force_cuboid:
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
            if not Polygon(xy2d).is_valid:
                print('Fail to generate valid general layout!! '
                        'Generate cuboid as fallback.',
                        file=sys.stderr)
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = horizonnet_utils.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    cor = np.hstack([cor, horizonnet_utils.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out

def find_corresponding_img_name(name, imgs):
    for img in imgs:
        if img.__contains__(name):
            return img

def scan_environment(imgs):
    device = torch.device('cpu')
    rgb_raw_imgs = []
    horizon_model = model_objs['horizon_net']
    horizon_model.eval()
    with torch.no_grad():
        for img in imgs:
            k = os.path.split(img)[-1][:-4]
            img_pil = Image.open(img)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])
            cor_id, z0, z1, vis_out = inference(net=horizon_model, x=x, device=device,
                                                flip=False, rotate=[],
                                                visualize=True, force_cuboid=False,
                                                force_raw=False,
                                                min_v=None, r=0.05)
            if vis_out is not None:
                vis_path = os.path.join('./', k + '.raw.png')
                rgb_raw_imgs.append(vis_path)
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                    .resize((vw//2, vh//2), Image.LANCZOS)\
                        .save(vis_path)

    W = 0
    H = 0
    L = 0
    V = 0

    for img in rgb_raw_imgs:
        image = cv2.imread(img)
        image = cv2.resize(image, (640, 510))
        gt_image = image[:15, :]
        rows, cols, _ = gt_image.shape

        lower_b = np.array([220, 220, 220])
        upper_b = np.array([255, 255, 255])

        mask = cv2.inRange(gt_image, lower_b, upper_b)
        idx = np.nonzero(mask)
        idx = np.transpose(idx)
        sorted_arr = np.sort(idx[:, 1])
        left, right = find_vanishing_points(sorted_arr)
        height_col = left + 30

        dense_model = model_objs['dense_depth']
        orig_img = os.path.split(img)[-1][:-8] + '.png'
        corr_img = find_corresponding_img_name(orig_img, imgs)
        orig_imgs = load_images([corr_img])
        output = predict(dense_model, orig_imgs)
        output = dense_model_scale_up(2, output)
        viz = display_images(output.copy())
        #matplotlib.image.imsave(os.path.split(img)[-1][:-8] + '_depth.png', viz)
        depth_img = output[0,:,:,0]
        left_arr = depth_img[:, left+30]
        right_arr = depth_img[:, right-10]
        left_plateaus = find_plateaus(left_arr)
        right_plateaus = find_plateaus(right_arr)

        
        
        lp = find_larger_plateau(left_plateaus)
        rp = find_larger_plateau(right_plateaus)
        print("PLATEAU ON THE LEFT: ")
        print(lp)
        print("PLATEAU ON THE RIGHT: ")
        print(rp)
        height_cor = lp
        if abs(lp[1] - lp[0]) < abs(rp[1] - rp[0]):
            print("using right for height")
            height_col = right -10
            height_cor = rp
        plt.imshow(depth_img)
        plt.savefig(os.path.split(img)[-1][:-8] + '_depth.png')
        #plt.show()
        #calculating width of room
        u1 = left
        v1 = height_cor[0]
        z1 = depth_img[v1, u1]
        p1 = calc_real_world_coor(u1, v1, z1)
        u2 = right
        v2 = height_cor[0]
        z2 = depth_img[v2, u2]
        p2 = calc_real_world_coor(u2, v2, z2)

        w = calc_3d_distance(p1, p2)
        if w > W:
            W = w

        print(orig_img)
        #calculating height of room
        u1 = height_col
        v1 = height_cor[0]
        z1 = np.transpose(depth_img)[u1, v1]
        p1 = calc_real_world_coor(u1, v1, z1)

        print(str(u1) + "\t" + str(v1))
        u2 = height_col
        v2 = height_cor[1]
        z2 = np.transpose(depth_img)[u2, v2]
        p2 = calc_real_world_coor(u2, v2, z2)

        print(str(u2) + "\t" + str(v2))
        h = calc_3d_distance(p1, p2)
        print("HEIGHT: " + str(h))
        print(p1)
        print(p2)
        if h > H:
            H = h

        L += calc_length(W/2, depth_img[height_cor[0], 240] * 10)

    V = W*H*L
    return W, H, L, V
