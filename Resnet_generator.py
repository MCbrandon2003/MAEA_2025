import torch
import numpy as np
from DIRT.domain_gen.solver_stargan import Solver
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
from train_Stargan import get_config
from dataset_utils import get_preprocess
import matplotlib.pyplot as plt
from M_domain_dataset import MultiDomainImageFolder
from M_domain_dataset import PairedImageFolder
import os
from train import Norm,Inverse_Norm,project
from CDA.generators import GeneratorResnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(dataloader,epochs,steps_per_epoch,classifier,G,filename="Resnet_Generator.pth"):
    G.train()
    G.to(device)
    
    optimizer = torch.optim.Adam(G.parameters(),lr=1e-5)
    
    if os.path.exists(filename):
        print(f"loading from {filename}")
        G.load_state_dict(torch.load(filename))
    
    classifier.eval()
    classifier.to(device)
    
    for param in classifier.parameters():
        param.requires_grad = False
    
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    
    for epoch in range(epochs):
        step = 0
        for x,label in dataloader:
            x = x.to(device);label = label.to(device)
            
            adv_x = G(x)
            adv_x = project(x,adv_x)
            
            pred_adv = classifier.forward(adv_x)
            pred_clean = classifier.forward(x)
            loss = -F.cross_entropy(pred_adv - pred_clean,label)
            
            loss.backward()
            optimizer.step()
            suc_rate = (pred_adv.argmax(dim=1)!=label).sum() / x.shape[0]
            print(f"epoch {epoch} step {step}: loss:{-loss} suc_rate:{suc_rate}")
            step += 1
            if(step>=steps_per_epoch):
                break
        torch.save(G.state_dict(),filename)
        print(f"Model saved to {filename}")
        
def test_CDTA(filename,G):
    names = ["birds_400","comic_books","food_101","oxford_102_flower"]
    paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/valid","../../../CDTA_datasets/comic_books/test","../../../CDTA_datasets/food_101/food-101/images","../../../CDTA_datasets/oxford_flower_102/dataset/valid"]
    ckpt_paths = ["../../../CDTA_models/BIRDS-400/inception_v3.pth.tar","../../../CDTA_models/Comic Books/inception_v3.pth.tar",
                  "../../../CDTA_models/Food-101/inception_v3.pth.tar","../../../CDTA_models/Oxford 102 Flower/inception_v3.pth.tar"]
    if os.path.exists(filename):
        G.load_state_dict(torch.load_dict(filename))
    preprocess = get_preprocess()
    
    G.to(device)
    G.eval()
    for param in G.parameters():
      param.requires_grad = False
    
    suc_rate = []
    for i,name in enumerate(names):  
      dataset = datasets.ImageFolder(paths[i],transform = preprocess)
      dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=3)
      
      num_classes = len(os.listdir(paths[i]))
      classifier = torchvision.models.inception_v3(num_classes=num_classes,aux_logits=False)
      classifier.eval()
      classifier.load_state_dict(torch.load(ckpt_paths[i]))
      classifier.to(device)
      for param in classifier.parameters():
        param.requires_grad = False
      
      fool = 0; total = 0; step = 0;
      for x,label in dataloader:
        x = x.to(device); label = label.to(device)
        adv_x = G(x)
        adv_x = project(x,adv_x)
        pred_adv = classifier.forward(adv_x)
        pred_clean = classifier.forward(x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step>=10:
          print(f"{name} fool_rate:{fool/total}")
          suc_rate.append((fool/total).cpu().detach().numpy())
          break
    print(np.mean(suc_rate))
if __name__ == "__main__":
    preprocess = get_preprocess()
    classifier = torchvision.models.inception_v3(pretrained=True)
    dataset = datasets.ImageFolder("../../../imagenet_domains/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=4)
    filename = "mae_models/Resnet_Generator.pth"
    G = GeneratorResnet()
    train(dataloader,30,20,classifier,G,filename)

