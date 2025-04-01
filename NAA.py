import random
import os
random.seed(0)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pickle
from tqdm import tqdm
from dataset_utils import get_preprocess,get_dataloader
from model_utils import prepare_mae_model,prepare_classifier
import timm
import cv2 as cv
from model_utils import VGG16
import scipy.stats as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, Pad

def Inverse_Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1))\
                  .float().to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1))\
                  .float().to(device)
  return x*imagenet_std + imagenet_mean

def Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1))\
                  .float().to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1))\
                  .float().to(device)
  return (x-imagenet_mean)/imagenet_std

def project(x0,adv_x,eps=16./255.):
    x0 = Inverse_Norm(x0)
    adv_x = Inverse_Norm(adv_x)
    pert = torch.clamp((adv_x-x0),-eps,eps)
    adv_x = torch.clamp((pert+x0),0,1)
    return Norm(adv_x)

def NAA_map(X,y,model,layer_name,N=30,drop_rate=0.3):

    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o        
    
    def get_mid_grad(m, i, o):
        global mid_grad
        mid_grad = o       
    if layer_name=="15":
      feature_layer = model._modules.get("features")._modules.get(layer_name)
    else:
      feature_layer = model._modules.get(layer_name)
    
    h = feature_layer.register_forward_hook(get_mid_output)
    h2 = feature_layer.register_full_backward_hook(get_mid_grad)
  
    agg_grad = 0
    for i in range(N):       
        X_step = i/N * X.clone()
        X_step.requires_grad = True
        output_random = model(X_step)        
        loss = 0
        for batch_i in range(X.shape[0]):
            loss += output_random[batch_i][y[batch_i]]        
        model.zero_grad()
        loss.backward()        
        agg_grad += mid_grad[0].clone().detach()    
    for batch_i in range(X.shape[0]):
        agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
    h2.remove()   
    return agg_grad


def NAA_attack(dataloader,model,target_model,layer_name):
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    model.to(device); 
    model.eval(); 

    momentum = 0
    fool = 0; total = 0
    for x,label in dataloader:
        x = x.to(device)
        label = label.to(device)
        Map = NAA_map(x,label,model,layer_name)
        model._modules.get(layer_name).register_forward_hook(get_mid_output)
        adv_x = x.clone()
        for i in range(10):         
          adv_x.requires_grad = True
          model.forward(adv_x)   
          loss = (Map*mid_output).sum()
          loss.backward()
          grad = adv_x.grad
          with torch.no_grad():
            momentum =  momentum + grad / torch.sum(torch.abs(grad),dim=[1,2,3])
            adv_x = adv_x - 0.274*torch.sign(momentum)/10
            adv_x = project(x,adv_x)
        return x




