import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import glob, os
import json
import argparse

from functions import load_checkpoint, process_image, predict, label_mapping, plotting

parser = argparse.ArgumentParser(description='Predict flower app')

parser.add_argument('path_to_image', action="store")
parser.add_argument('checkpoint', action="store")

parser.add_argument('--category_names', action="store", dest="cat_names", default = "cat_to_name.json")
parser.add_argument('--top_k', action="store", dest="top_k", type=int, default = 1)
parser.add_argument('--gpu', action="store", dest="gpu", default = "gpu")

parse_args = parser.parse_args()

# 'flowers/test/1/image_06752.jpg'
image_path = parse_args.path_to_image
# 'checkpoint.pth'
save_path = parse_args.checkpoint

cat_names = parse_args.cat_names
top_k = parse_args.top_k
gpu = parse_args.gpu


#load saved model
model, device, output_units, criterion, optimizer = load_checkpoint(save_path,gpu)

#category names
cat_to_name = label_mapping(cat_names)

#predict
plotting(image_path,model,top_k,cat_to_name)


              