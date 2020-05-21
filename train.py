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

from functions import data_load, define_model, model_train, save_checkpoint

parser = argparse.ArgumentParser(description='train model app')

parser.add_argument('data_dir', action="store")

parser.add_argument('--save_dir', action="store", dest="save_dir", default = "checkpoint.pth")
parser.add_argument('--arch', action="store", dest="arch", default = "densenet121")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=int, default = 0.003)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default = 1024)
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default = 2)
parser.add_argument('--gpu', action="store", dest="gpu", default = "gpu")

parse_args = parser.parse_args()

data_dir = parse_args.data_dir

save_path = parse_args.save_dir
arch = parse_args.arch
learning_rate = parse_args.learning_rate
hidden_units = parse_args.hidden_units
epochs = parse_args.epochs
gpu = parse_args.gpu


train_datasets, valid_datasets, test_datasets, trainloader, validloader, testloader =  data_load(data_dir)

model, device, output_units, criterion, optimizer = define_model(arch, learning_rate, hidden_units, gpu)

model_train(trainloader, validloader, testloader, epochs, model, device, output_units, criterion, optimizer)

save_checkpoint(save_path, arch, hidden_units, learning_rate, epochs)


