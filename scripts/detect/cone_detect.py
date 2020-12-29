#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:57:50 2020

@author: shijiliu
"""


import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from numpy import random
import os
import logging
import math
#from generate_path import sort_path_pt
#from common import Conv, DWConv

# import from general.py, which is copied from yolov3/utils/general.py
from general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path

logger = logging.getLogger(__name__)


# following codes are copied from models/common.py
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# following codes are copied from yolov3/utils/torch_utils
def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    '''
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity
    '''
    cuda = False if cpu_request else torch.cuda.is_available()
    print(cuda)
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# end codes from torch_tuils

# the following codes are copied from yolov3/utils/plots.py
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# end codes from plots.py

# following codes are copied from yolov3/utils/experimental.py
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble
# end codes from yolov3/utils/experimental.py

# the following codes are copied from yolov3/utils/datasets.py
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def sort_path_pt_few(points):
    distance_mat = np.zeros((len(points),len(points)))
    
    # calculate distance
    for ii in range(len(points)):
        for jj in range(1,len(points)):
            d = np.linalg.norm(points[ii] - points[jj])
            distance_mat[ii][jj] = d
            distance_mat[jj][ii] = d
    
    # 
    



class DetectionClass(object):
    def __init__(self, weight_file = "best.pt", img_size = 640, device = 'cuda:0', conf = 0.25, iou_thres = 0.45, classes = None, augment = False, agnostic_nms = False):
        self.weight_file = weight_file
        self.conf = 0.25 # confidence for predicition
        self.iou_thres = iou_thres # IOU threshold fors NMS
        self.classes = classes
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        
        # get cuda device
        self.device = select_device('cuda')
        
        # check whether half precision can be applied
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        # load the model
        self.model = attempt_load(self.weight_file).to(self.device)
        
        self.img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size,the image will be img_size * img_size, e.g. 640*640
        if self.half:
            self.model.half()
        
        # get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[0,0,255] if name == 'r' else [255,0,0]  for name in self.names] # (B,G,R)
        
    def detect(self,img0):
        # img read by cv2.imread(), in BGR format
        
        t0 = time.time()
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # change img from numpy array into torch tensor
        img = torch.from_numpy(img).to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0] # path img through model
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        
        for i, det in enumerate(pred): # detections per image, here only one image
            im0 = img0
            gn = torch.tensor(im0.shape)[[1,0,1,0]] # normalization gain whwh
            if len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                #print(det)
                
                for det0 in reversed(det):
                    
                    xyxy = det0[0:4]
                    conf = det0[4]
                    classes = det0[5]
                    
                    label = ('%s %.2f') % (self.names[int(classes)], conf)
                    # add bounding box to image
                    plot_one_box(xyxy, im0,label=label, color=self.colors[int(classes)], line_thickness=3)
        
        
        print('Done. (%.3fs)' % (time.time() - t0))            
                
            
        return im0
    
    
    def get_central_pts(self, img0):
        # img read by cv2.imread(), in BGR format
        
        #t0 = time.time()
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # change img from numpy array into torch tensor
        img = torch.from_numpy(img).to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0] # pass img through model
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        
        red_pts = []
        blue_pts = []
        
        for i, det in enumerate(pred): # detections per image, here only one image
            im0 = img0
            gn = torch.tensor(im0.shape)[[1,0,1,0]] # normalization gain whwh
            if len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                #print(det)

                for det0 in reversed(det):
                    #print(det0)
                    v1 = int(det0[1])
                    v2 = int(det0[3])
                    u1 = int(det0[0])
                    u2 = int(det0[2])
                    classes = det0[5]
                    
                    u_center = (u1 + u2) / 2
                    v_center = (v1 + v2) / 2
                    
                    if classes.cpu() == 0:
                        red_pts.append(np.array([u_center,v_center]))
                    else:
                        blue_pts.append(np.array([u_center,v_center]))
        

        #print('Done. (%.3fs)' % (time.time() - t0))            
                
            
        return red_pts, blue_pts
    
    
    
    def get_cone_image(self,img0):
        '''
        1. use trained yolo model to find bounding box of red/blue cones;
        2. create a numpy array with self.img_size * self.img_size, initialized to 0.5. i.e. 0.5 * np.ones((self.img_size,self.img_size))
        3. change the values in the array: red cone region 0, blue cone region 1

        Parameters
        ----------
        img0 : image
            An image read by cv2, in BGR format

        Returns
        -------
        cone_array : np.array()
            the array representation of the cone detection result

        '''
        #t0 = time.time()
        
        # create the array
        cone_array = 0.5 * np.ones((self.img_size,self.img_size))
        
        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # change img from numpy array into torch tensor
        img = torch.from_numpy(img).to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0] # path img through model
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()
        
        for i, det in enumerate(pred): # detections per image, here only one image
            #gn = torch.tensor(im0.shape)[[1,0,1,0]] # normalization gain whwh
            if len(det):
                # rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                #print(det)
                
                for det0 in reversed(det):
                    
                    # note that the coordinate system in an array is different from
                    # the one used in image, see https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data#2-create-labels
                    x1 = int(det0[1])
                    x2 = int(det0[3] + 1)
                    y1 = int(det0[0])
                    y2 = int(det0[2] + 1)
                    
                    
                    classes = det0[5]
                    
                    # change value in cone_array
                    cone_array[x1:x2,y1:y2] = classes.cpu() # classes: red == 0.0, blue == 1.0
        
        
        #print('Done. (%.3fs)' % (time.time() - t0))     
        return cone_array
    
    def visualize_cone_img(self, cone_image):
        img = np.zeros((self.img_size,self.img_size,3))
        for ii in range(self.img_size):
            for jj in range(self.img_size):
                if cone_image[ii][jj] == 0:
                    img[ii][jj] = [0,0,255]
                elif cone_image[ii][jj] == 1:
                    img[ii][jj] = [255,0,0]
        
        return img

if __name__ == '__main__':
    cone_detection = DetectionClass()
    img = cv2.imread('test_img.jpg')
    red_pts, blue_pts = cone_detection.get_central_pts(img)
    print(red_pts)
    #detected_img = cone_detection.detect(img)
    #cv2.imshow('detection result',detected_img)
    #cv2.imwrite('detected.jpg',detected_img)
    #t0 = time.time()
    #n = 10
    #for ii in range(n):
    #    cone_array = cone_detection.get_cone_image(img)
    #print("avg time: %f" % ( (time.time() - t0) / n) )     
    #visual_cone_array = cone_detection.visualize_cone_img(cone_array)
    #cv2.imwrite('visual_cone_array.jpg',visual_cone_array)

