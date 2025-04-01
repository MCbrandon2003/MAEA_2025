import torch
from torch import nn
import torch.nn.functional as F
from mae_pytorch import models_vit
from mae_pytorch.util import misc
from dataset_utils import get_preprocess
import argparse
import torchvision
from attack_utils import TI_attack
from dataset_utils import SVHNDataset,CIFAR100Dataset
import os
import timm
from EfficientNetB7 import get_args

def get_dataset(name):
  preprocess = get_preprocess()
  names = ["birds_400","food_101","oxford_102_flower","stanford_cars"]
  paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/train",
          "../../../CDTA_datasets/food_101/food-101/images",
          "../../../CDTA_datasets/oxford_flower_102/dataset/train",
          "../../../CDTA_datasets/cars_train/cars_train"]
  for i,n in enumerate(names):
    if name == n:
      return torchvision.datasets.ImageFolder(paths[i],transform=preprocess)
  if name == "SVHN":
    dataset = SVHNDataset()
  if name == "CIFAR100":
    dataset = CIFAR100Dataset()
  return dataset

def get_dataset_path(name):
  preprocess = get_preprocess()
  names = ["birds_400","food_101","oxford_102_flower","stanford_cars"]
  paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/train",
          "../../../CDTA_datasets/food_101/food-101/images",
          "../../../CDTA_datasets/oxford_flower_102/dataset/train",
          "../../../CDTA_datasets/cars_train/cars_train"]
  for i,n in enumerate(names):
    if name == n:
      return paths[i]

def get_args():
  parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
  parser.add_argument('--resume', default='mae_pytorch/mae_finetuned_vit_large.pth',
                        help='resume from checkpoint')
  parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
  parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
  parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
  parser.add_argument('--global_pool', action='store_true')
  parser.set_defaults(global_pool=True)
  args = parser.parse_args()
  return args

def get_model(model_type,dataset):
  if dataset == "CIFAR100":
    num_classes = 100
  elif dataset == "SVHN":
    num_classes = 10
  else:
    path = get_dataset_path(dataset)
    num_classes = len(os.listdir(path))
  if model_type == "EfficientNetB7":   
    EB7 = timm.create_model("tf_efficientnet_b7_ns",num_classes=num_classes)
    EB7.load_state_dict(torch.load(f"SAM_models/{model_type}_{dataset}.pth"))
    return EB7
  if model_type == "Resnet152":
    resnet = torchvision.models.resnet152(num_classes=num_classes)
    resnet.load_state_dict(torch.load(f"SAM_models/{model_type}_{dataset}.pth"))
    return resnet
  if model_type == "Resnet50":
    resnet = torchvision.models.resnet50(num_classes=num_classes)
    resnet.load_state_dict(torch.load(f"SAM_models/{model_type}_{dataset}.pth"))
    return resnet
  if model_type == "ViT":
    ViT = models_vit.__dict__["vit_large_patch16"](
          num_classes=1000,
          drop_path_rate=0.1,
          global_pool=True,
      )
    args = get_args()
    ViT.head = torch.nn.Linear(ViT.head.in_features, num_classes)
    ViT.load_state_dict(torch.load(f"SAM_models/{model_type}_{dataset}.pth"))
    return ViT








