import torch
import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch import nn
import argparse
import torchvision.datasets as datasets
import torchvision
import torch.nn.functional as F
from torchvision import transforms as T
from torch.optim import Adam
from model_utils import prepare_mae_model,prepare_classifier
from torchvision.datasets.food101 import Path
from dataset_utils import get_dataloader
import matplotlib.pyplot as plt
from attack_utils import Norm,Inverse_Norm
import os
from model_utils import MAE_Classifier
import timm
from model_utils import VGG16
from torchvision import transforms
from sam.example.utility.bypass_bn import enable_running_stats,disable_running_stats
from sam.example.model.smooth_cross_entropy import smooth_crossentropy
from sam.sam import SAM
from mae_pytorch import models_vit
from mae_pytorch.util import misc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def FGSM(model, img, label):
    img.requires_grad = True
    output = model(img)
    loss = torch.nn.functional.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    x = Inverse_Norm(img)
    # Generate perturbation
    perturbation = img.grad.data.sign()

    # Get random epsilon from N(0,8) distribution
    epsilon = abs(np.random.normal(0, 8)) / 255.

    # Apply perturbation
    perturbed_img = x + epsilon * perturbation

    # Adding clipping to maintain [0,1] range
    perturbed_img = torch.clamp(perturbed_img, 0, 1)

    perturbed_img = Norm(perturbed_img)
    return perturbed_img

def train(dataloader, epochs, model, filename, adv = False):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)  # define an optimizer for the "sharpness-aware" update
    model.to(device)
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        print(f"loading from {filename}")

    for epoch in range(epochs):
        step = 0
        for x, label in dataloader:
            x = x.to(device); label = label.to(device).squeeze()
            if adv == True:
              adv_x = FGSM(model,x,label)
            

            predictions = model(x)
            adv_pred = model(adv_x)

            loss = F.cross_entropy(predictions, label) + 0.3 * F.cross_entropy(adv_pred, label)
            loss.mean().backward()
            optimizer.step()
    
            _, predicted = torch.max(predictions.data, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            print(f"Step {step}, Loss: {loss.mean().item():.3f}, Accuracy: {accuracy:.3f}")
            
            step += 1
            if step % 100 == 0:
                torch.save(model.state_dict(), filename)
                print(f"model saved to {filename}")
            if step >= 500:
                break
        torch.save(model.state_dict(), filename)
        print(f"model saved to {filename}")

            
def set_up_train(model:str,dataset:str,epochs=2):
  assert model in ["EfficientNetB7","Resnet152","Resnet50","ViT"]
  assert dataset in ["oxford_flower_102","stanford_cars","SVHN","CIFAR100"]

  num_dict = {"oxford_flower_102":102,"stanford_cars":196,
              "SVHN":10,"CIFAR100":100}

  num_classes = num_dict[dataset]
  dataloader = get_dataloader(dataset)
  filename = f"Adv_models/{model}_{dataset}.pth"
  if model == "EfficientNetB7":
    net = timm.create_model("tf_efficientnet_b7_ns",pretrained=True)
    net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)
    net.train()
    net.to(device)
  elif model == "ViT":
    net = models_vit.__dict__["vit_large_patch16"](
          num_classes=1000,
          drop_path_rate=0.1,
          global_pool=True,
      )
    net.train()
    args = get_args()
    misc.load_model(args,net,None,None)
    net.head = torch.nn.Linear(net.head.in_features, num_classes)
    net.to(device)
  elif model == "Resnet152":
    net = torchvision.models.resnet152(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    net.train()
    net.to(device)
  elif model == "Resnet50":
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    net.train()
    net.to(device)
  net.load_state_dict(torch.load(f"SAM_models/{model}_{dataset}.pth"))
  print(f"Loading from SAM_models/{model}_{dataset}.pth")
  train(dataloader,epochs,net,filename,adv=True)
        
if __name__ =='__main__':
  for model in ["EfficientNetB7","Resnet152","Resnet50","ViT"]:
    for dataset in ["oxford_flower_102","stanford_cars","SVHN","CIFAR100"]:
      if os.path.exists(f"Adv_models/{model}_{dataset}.pth"):
        print(f"Adv_models/{model}_{dataset}.pth exists")
        continue
      set_up_train(model,dataset)











