import os, sys, torch, json, re
import torch as ch
import numpy as np
from PIL import Image
from torchvision import transforms
from robustness.tools.label_maps import CLASS_DICT
from robustness import datasets
from robustness import data_augmentation as da
from helpers.gen_helpers import find_duplicates
from utils import parallelfolder, renormalize
from utils.custom_vgg import vgg16_bn, vgg16
from utils.custom_resnet import resnet18, resnet50
from utils.places365_names import class_dict

def get_default_paths(dataset_name, eps, arch='vgg16'):
    if dataset_name == 'ImageNet':
        data_path = '/tmp/imagenet'
        label_map = CLASS_DICT['ImageNet']
        
        if arch.startswith('clip'):
            model_path = None
            model_class = None
            arch = arch
        elif arch == 'resnet50':
            model_path = '/tmp/resnet/checkpoint.pt'
            model_class, arch = resnet50(), 'resnet50'
        else:
            model_path = '/tmp/vgg/checkpoint.pt'
            model_class, arch = vgg16_bn(), 'vgg16_bn'
        
    else:
        data_path = '/tmp/places'
        label_map = class_dict
        
        if arch.startswith('vgg16'):
            model_path = '/tmp/vgg/checkpoint.pt'
            model_class, arch = vgg16(num_classes=365), 'vgg16'
        elif arch == 'resnet18':
            model_path = '/tmp/resnet/checkpoint.pt'
            model_class, arch = resnet18(num_classes=365), 'resnet18'
        
        
    return data_path, model_path, model_class, arch, label_map
