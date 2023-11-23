# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import glob
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print('ROOT = ', ROOT)
from obj_detect_models.yolo_v5.models.experimental import attempt_load
from obj_detect_models.yolo_v5.utils.general import (check_suffix, non_max_suppression, xyxy2xywh, scale_coords)
from obj_detect_models.yolo_v5.utils.torch_utils import time_sync
from obj_detect_models.yolo_v5.utils.augmentations import letterbox
#from droidbot.adapter.easyocr.easyocr import *
# weights=ROOT / 'obj_detect_models/yolo_v5/best.pt'  # model.pt path(s)
# weights=ROOT / 'obj_detect_models/yolo_v5/best_new.pt'  # model.pt path(s)
weights=ROOT / 'obj_detect_models/yolo_v5/best_old_11.pt'  # model.pt path(s)
#source=ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
imgsz=640  # inference size (pixels)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=False  # show results
save_txt=False # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=True  # class-agnostic NMS
augment=False  # augmented inference
visualize=False # visualize features
update=False  # update all obj_detect_models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference

#source = str(source)
#print('source = ', source)
#print('device = ', device)

# Load model
w = str(weights[0] if isinstance(weights, list) else weights)
classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
check_suffix(w, suffixes)  # check weights have acceptable suffix
pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#imgsz = check_img_size(imgsz, s=stride)  # check image size

path = 'img.jpg'
#print('path = ', path)

def yolov5_detection(im0s, ui2_only_text, img_size=640, conf_thres=0.25, iou_thres=0.45, agnostic_nms=True, max_det=1000):
    #easyocr using
    # reader = Reader(['ko', 'en'], recog_network='ko_updateV9-3', gpu=True)
    # result = reader.readtext(im0s)
    # print('OOOOOOOOOCCCCCCCCCCCCCCCCRRRRRRRRRRRRRRRRRRRRRR = ', result)
    # ocr_labels = []
    # labels_str_ocr = ''
    # for (bbox, string, confidence) in result:
    #     #label = string
    #     accuracy = confidence
    #     ocr_labels.append(string)
    #     labels_str_ocr += string
    #print('yolov5_detection')
    dt, seen = [0.0, 0.0, 0.0], 0

    #im0s = cv2.imread(img_path)  # BGR
    h_org, w_org, ch_org = im0s.shape
    div_h = h_org / img_size
    div_w = w_org / img_size
    #print('h_org = {}, w_org = {}, ch_org = {}'.format(h_org, w_org, ch_org))
    assert im0s is not None, f'Image Not Found {path}'
    #print('img0 = ', im0s)

    # Padded resize
    img = letterbox(im0s, img_size, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    #print('img = ', img)

    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(img, augment=augment, visualize=False)[0]

    #print('pred = ========', pred)

    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #print('pred nms = ========', pred)
    dt[2] += time_sync() - t3
    labels_str = ''
    # Process predictions
    results = []
    results_boxes = []
    for i, det in enumerate(pred):  # per image
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        if len(det):
            # Write results
            signature_list = []
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                #print('label = ', label)
                #print('label l = ', type(label))

                #print('b t = ', float(b[0][0]))
                x1 = float(xyxy[0])
                y1 = float(xyxy[1])
                x2 = float(xyxy[2])
                y2 = float(xyxy[3])

                results.append([label, x1, y1, x2, y2])
                #labels_str += label + "x1=" + str(int(x1)) + "y1=" + str(int(y1)) + "x2=" + str(int(x2)) + "y2=" + str(int(y2)) + "$$$"

                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                resource_id = ''
                #if y > 20:
                labels_str += label + "x=" + str(x) + "y=" + str(y) + "w=" + str(width) + "h=" + str(height)

                signature = "label=" + label + "x=" + str(x) + "y=" + str(y) + "w=" + str(width) + "h=" + str(height)
                # if num < 5:
                #     signature_list.append(label)
                #results_boxes.append((x, y, width, height, label, signature))
                #results_boxes.append((x, y, width, height, label, ui2_only_text))
                results_boxes.append((x, y, width, height, label, ui2_only_text))
                accuracy = "{:.4f}".format(conf)
                #print('accuracy = ', accuracy)
    # for ui in ui2_only_text:
    #     x1 = float(ui[1].split('[')[1].split(',')[0])
    #     y1 = float(ui[1].split('[')[1].split(',')[1].split(']')[0])
    #     x2 = float(ui[1].split('[')[2].split(',')[0])
    #     y2 = float(ui[1].split('[')[2].split(',')[1].split(']')[0])
    #     x = int(x1)
    #     y = int(y1)
    #     width = int(x2 - x1)
    #     height = int(y2 - y1)
    #
    #     results_boxes.append((x, y, width, height, 'Text', ui2_only_text))

    #print('results_boxes = ', results_boxes)
    return results_boxes

def yolov5_detection_wo_ui2(img_path, img_size=640, conf_thres=0.25, iou_thres=0.45, agnostic_nms=True, max_det=1000):
    #easyocr using
    # reader = Reader(['ko', 'en'], recog_network='ko_updateV9-3', gpu=True)
    # result = reader.readtext(im0s)
    # print('OOOOOOOOOCCCCCCCCCCCCCCCCRRRRRRRRRRRRRRRRRRRRRR = ', result)
    # ocr_labels = []
    # labels_str_ocr = ''
    # for (bbox, string, confidence) in result:
    #     #label = string
    #     accuracy = confidence
    #     ocr_labels.append(string)
    #     labels_str_ocr += string
    #print('yolov5_detection')
    dt, seen = [0.0, 0.0, 0.0], 0

    im0s = cv2.imread(img_path)  # BGR
    h_org, w_org, ch_org = im0s.shape
    div_h = h_org / img_size
    div_w = w_org / img_size
    #print('h_org = {}, w_org = {}, ch_org = {}'.format(h_org, w_org, ch_org))
    assert im0s is not None, f'Image Not Found {path}'
    #print('img0 = ', im0s)

    # Padded resize
    img = letterbox(im0s, img_size, stride=stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    #print('img = ', img)

    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(img, augment=augment, visualize=False)[0]

    #print('pred = ========', pred)

    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #print('pred nms = ========', pred)
    dt[2] += time_sync() - t3
    labels_str = ''
    # Process predictions
    results = []
    results_boxes = []
    for i, det in enumerate(pred):  # per image
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        if len(det):
            # Write results
            signature_list = []
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                #print('label = ', label)
                #print('label l = ', type(label))

                #print('b t = ', float(b[0][0]))
                x1 = float(xyxy[0])
                y1 = float(xyxy[1])
                x2 = float(xyxy[2])
                y2 = float(xyxy[3])

                results.append([label, x1, y1, x2, y2])
                #labels_str += label + "x1=" + str(int(x1)) + "y1=" + str(int(y1)) + "x2=" + str(int(x2)) + "y2=" + str(int(y2)) + "$$$"

                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                resource_id = ''
                #if y > 20:
                labels_str += label + "x=" + str(x) + "y=" + str(y) + "w=" + str(width) + "h=" + str(height)

                signature = "label=" + label + "x=" + str(x) + "y=" + str(y) + "w=" + str(width) + "h=" + str(height)
                # if num < 5:
                #     signature_list.append(label)
                #results_boxes.append((x, y, width, height, label, signature))
                #results_boxes.append((x, y, width, height, label, ui2_only_text))
                results_boxes.append((x, y, width, height, label))
                accuracy = "{:.4f}".format(conf)
                # print('accuracy = ', accuracy)
    # for ui in ui2_only_text:
    #     x1 = float(ui[1].split('[')[1].split(',')[0])
    #     y1 = float(ui[1].split('[')[1].split(',')[1].split(']')[0])
    #     x2 = float(ui[1].split('[')[2].split(',')[0])
    #     y2 = float(ui[1].split('[')[2].split(',')[1].split(']')[0])
    #     x = int(x1)
    #     y = int(y1)
    #     width = int(x2 - x1)
    #     height = int(y2 - y1)
    #
    #     results_boxes.append((x, y, width, height, 'Text', ui2_only_text))

    #print('results_boxes = ', results_boxes)
    return results_boxes


def draw_bbox(img_path, results_boxes):
    img = cv2.imread(img_path)
    boxed_img = img.copy()

    for coordinate in results_boxes:
        cv2.rectangle(boxed_img, (coordinate[0], coordinate[1]),
                      (coordinate[2] + coordinate[0], coordinate[3] + coordinate[1]), (0, 0, 255), 2)

    cv2.imshow("boxed image", boxed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

jpg_list = glob.glob("./input/test_images/*.jpg")
for img_path in jpg_list:
    res = yolov5_detection_wo_ui2(img_path=img_path, img_size=imgsz)
    print(f"{res}")
    draw_bbox(img_path=img_path, results_boxes=res)
