import random
import os
random.seed(0)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import torchvision.datasets as datasets
from dataset_utils import get_preprocess,get_dataloader
from train import project,Norm,Inverse_Norm
from CDA.generators import GeneratorResnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_CDA(filename):
    names = ["birds_400","comic_books","food_101","oxford_102_flower"]
    paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/valid","../../../CDTA_datasets/comic_books/test","../../../CDTA_datasets/food_101/food-101/images","../../../CDTA_datasets/oxford_flower_102/dataset/valid"]
    ckpt_paths = ["../../../CDTA_models/BIRDS-400/inception_v3.pth.tar","../../../CDTA_models/Comic Books/inception_v3.pth.tar",
                  "../../../CDTA_models/Food-101/inception_v3.pth.tar","../../../CDTA_models/Oxford 102 Flower/inception_v3.pth.tar"]
    netG = GeneratorResnet()
    netG.load_state_dict(torch.load(f"CDA/saved_models/{filename}"))

    suc_rate = []
    preprocess = get_preprocess()
    netG.to(device)
    netG.eval()
    for param in netG.parameters():
      param.requires_grad = False

    for i,name in enumerate(names):  
      dataset = datasets.ImageFolder(paths[i],transform = preprocess)
      dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=4)
      
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
        adv_x = netG(Inverse_Norm(x).float())
        adv_x = project(x,Norm(adv_x))
        pred_adv = classifier.forward(adv_x)
        pred_clean = classifier.forward(x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step>=20:
          print(f"{name} fool_rate:{fool/total}")
          suc_rate.append((fool/total).cpu().detach().numpy())
          break
    print(np.mean(suc_rate))

def test_CDA_ImageNet(filename):
    netG = GeneratorResnet()
    netG.load_state_dict(torch.load(f"CDA/saved_models/{filename}"))

    suc_rate = []
    preprocess = get_preprocess()
    netG.to(device)
    netG.eval()
    net = torchvision.models.resnet152(pretrained=True)
    net.eval()
    net.to(device)
    for param in net.parameters():
      param.requires_grad = False
    for param in netG.parameters():
      param.requires_grad = False

    dataset = torchvision.datasets.ImageFolder("../../../imagenet-mini/val",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=30,shuffle=True,num_workers=2,prefetch_factor=4)
    fool = 0; total = 0; step = 0;
    for x,label in dataloader:
      x = x.to(device); label = label.to(device)
      adv_x = netG(Inverse_Norm(x).float())
      adv_x = project(x,Norm(adv_x))
      pred_adv = net(adv_x)
      pred_clean = net(x)
      fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
      total += (pred_clean.argmax(dim=1)==label).sum()
      step += 1
      if step>=20:
        print(f"fool_rate:{fool/total}")
        suc_rate.append((fool/total).cpu().detach().numpy())
        break

if __name__ == '__main__':
    test_CDA("netG_-1_img_vgg16_imagenet_0_rl.pth")






