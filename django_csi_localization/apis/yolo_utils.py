import itertools
from keras.models import load_model
import argparse
import time
from pathlib import Path
from apis.utils import model_objs
from apis import utils
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json
from models.experimental import attempt_load
from apis.yolo_datasets import LoadStreams, LoadImages
from apis.yolo_general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from apis.yolo_plots import plot_one_box
from apis.yolo_torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import open3d as o3d
import math
import numpy as np
from scipy.spatial.distance import pdist

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

def detect(weights=['./models/yolov7.pt'], source='../input/test_1', img_size=640, conf_thres = 0.25, iou_thres=0.45, device='cpu', write_json=True, save_txt=False, save_conf=False, project='runs/detect', name='exp', exist_ok=False, trace=False, view_img=False, save_img=False):

    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    print(save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    print(weights)
    # Load model
    print(device)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    #JSON return object
    ret = {}
    ret_arr = []
    num_obj = 0

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
#                model(img, augment=opt.augment)[0]
                model(img, augment=False)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#            pred = model(img, augment=opt.augment)[0]
            pred=model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if write_json:
                      obj_name = names[int(cls)]
                      np_xyxy = torch.tensor(xyxy).numpy()
                      cor = {
                        "x1": str(np_xyxy[0]),
                        "y1": str(np_xyxy[1]),
                        "x2": str(np_xyxy[2]),
                        "y2": str(np_xyxy[3])
                      }
                      detected_obj = {
                        "obj_name": obj_name,
                        "obj_cor": cor
                      }
                      ret_arr.append(detected_obj)
                      num_obj += 1

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(torch.tensor(xyxy))
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + names[int(cls)] + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(im0.shape)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    ret = {
      "objects": ret_arr,
      "count": str(num_obj)
    }
    json_ret = json.dumps(ret)
    print(f'Done. ({time.time() - t0:.3f}s)')
    return ret

def calc_density(imgs):
  total_v = 0
  for img in imgs:
    depth_img = gen_depth_map(model_objs['dense_depth'], img)
    detected_obj = detect(source=img)
    objects = detected_obj["objects"]
    for obj in objects:
      total_v += calc_obj_volume(obj, depth_img)
  return total_v
