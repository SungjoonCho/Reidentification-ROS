# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse

from yaml.tokens import TagToken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import time
import natsort
import shutil
########################
import sys

## ssd method detection
# sys.path.insert(0, 'person_detection_from_cctv_video')
# from person_detection_every_sec import crop

## yolo detection
sys.path.insert(0, 'yolov5')
from detect import detect_main
import cv2
########################

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
# test_dir = opt.test_dir #####


gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

# data transform
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

# remove runs/detect irectory
remove_file_path = './runs/detect'
try:
    shutil.rmtree(remove_file_path)
except:
    pass

######################### prepare datasets #########################

# directory_name = '/home/jskimlab/test_layumi/PRW-baseline/PRW/frame_test/'  # 이미지 모음 부를 경우

# for test1.mp4
video_path = './video/test1.mp4'
save_frame = './video/test1/'

# for test2.mp4
# video_path = './video/test2.mp4'
# save_frame = './video/test2/'


# extract frames
vidcap = cv2.VideoCapture(video_path)

total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
print('Total frames : ' ,total_frames, 'FPS : ', fps)

interval = 3 # 3초에 한 프레임

success,image = vidcap.read()

count = 1
success = True

while success:

  success,image = vidcap.read()
  if count % (fps * interval) == 0: 
    cv2.imwrite(os.path.join(save_frame,str(count)+'.jpg'), image)
  if total_frames-1 == count:                    
      break
  
  count += 1

print('Frame save complete')
vidcap.release()




########################## Crop each person and save in img file #########################
detect_main(save_frame) 

# 폴더 형태로 삽입 data transform and cropped person data load
dataset_path = './runs'
# file_list = natsort.natsorted(os.listdir(dataset_path))
# dataset_path = dataset_path + '/' + file_list[-1]

image_datasets = {'img': datasets.ImageFolder(dataset_path ,data_transforms)}
dataloaders = {'img': torch.utils.data.DataLoader(image_datasets['img'], batch_size=opt.batchsize,
                                            shuffle=False, num_workers=16)}

use_gpu = torch.cuda.is_available()


######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

# def get_id(img_path):
#     camera_id = []
#     labels = []
#     for path, v in img_path:
#         #filename = path.split('/')[-1]
#         filename = os.path.basename(path)
#         label = filename[0:4]
#         camera = filename.split('c')[1]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_id.append(int(camera[0]))
#     return camera_id, labels


######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)

if opt.PCB:
    model_structure = PCB(opt.nclasses)

#if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    #if opt.fp16:
    #    model = PCB_test(model[1])
    #else:
        model = PCB_test(model)
else:
    #if opt.fp16:
        #model[1].model.fc = nn.Sequential()
        #model[1].classifier = nn.Sequential()
    #else:
        model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


## modified
with torch.no_grad():
    img_feature = extract_feature(model,dataloaders['img'])
    
# Save to Matlab for check
result = {'img_f':img_feature.numpy()}
scipy.io.savemat('pytorch_result_test_yolov5_2.mat',result)