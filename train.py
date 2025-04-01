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
from dataset_utils import get_preprocess,data_augment
import matplotlib.pyplot as plt
import os
from model_utils import MAE_Classifier
from attack_utils import FIA_map,input_diversity
from NAA import NAA_map
from ensemble_training import train_FIA_ensemble
import timm
from model_utils import VGG16,prepare_classifier
from torchvision import transforms
from gradcam import generate_grad_cam,top_30_percent_gradcam
from ensemble_training import train_FIA_ensemble
from dataset_utils import SVHNDataset,CIFAR100Dataset,get_dataloader
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

class FIA_args:
  def __init__(self, model, layer_name, N=30, drop_rate=0.3):
    self.model = model
    self.layer_name = layer_name
    self.N = N
    self.drop_rate = 0.3

def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def Inverse_Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).to(device)
  return (x*imagenet_std + imagenet_mean).float()

def Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).to(device)
  return ((x-imagenet_mean)/imagenet_std).float()

def project(x0,adv_x,eps=16./255.):
    x0 = Inverse_Norm(x0)
    adv_x = Inverse_Norm(adv_x)
    pert = torch.clamp((adv_x-x0),-eps,eps)
    adv_x = torch.clamp(pert + x0,0,1)
    return Norm(adv_x).float()
        
def train(dataloader,epochs,steps_per_epoch,classifier,MAE_model=None,filename="Generator_trained_vanilla.pth",
          mask_ratio=0.5,freeze_encoder=False,F_args=None,DI=False,lda=1):
    classifier.eval()
    classifier.to(device)

    if MAE_model is None:
        MAE_model = prepare_mae_model(model_type="gan")
    MAE_model.shuffle = False
    MAE_model.to(device)
    optimizer = torch.optim.Adam(MAE_model.parameters(),lr=1e-5)  # define an optimizer for the "sharpness-aware" update

    if freeze_encoder == True:
        for blk in MAE_model.blocks:
            for param in blk.parameters():
                param.requries_grad = False

    MAE_model.shuffle = False
    
    if os.path.exists(filename):
        print(f"loading from {filename}")
        MAE_model.load_state_dict(torch.load(filename))
    
    for param in classifier.parameters():
        param.requires_grad = False
    
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    for epoch in range(epochs):
        step = 0
        for x,label in dataloader:
            optimizer.zero_grad()
            x = x.to(device);label = label.to(device).squeeze()
           
            # F_map = FIA_map(x,label,F_args.model,F_args.layer_name,F_args.N,F_args.drop_rate)
            
            _,decoded_token,_ = MAE_model(x,0)
            
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x, adv_x)
            if DI == True:
              adv_x = input_diversity(adv_x)

            if F_args is None:
              pred_adv = classifier.forward(adv_x)
              pred_clean = classifier.forward(x)
              loss = -F.cross_entropy(pred_adv - pred_clean,label)
            else:
              feature_layer = classifier._modules.get(F_args.layer_name)
              feature_layer.register_forward_hook(get_mid_output)
              pred_adv = classifier.forward(adv_x)
              loss = (F_map*mid_output).sum()
            fool = 0; total = 0
            fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
            total += (pred_clean.argmax(dim=1)==label).sum()
            fool_rate = fool / total
            loss.backward()
            optimizer.step()
            suc_rate = (pred_adv.argmax(dim=1)!=label).sum() / x.shape[0]
            
            print(f"epoch {epoch} step {step}:\
loss:{loss:.2f} fool_rate:{fool_rate:.3f} total:{total}")
            step += 1
            if(step>=steps_per_epoch):
                break
        torch.save(MAE_model.state_dict(),filename)
        print(f"Model saved to {filename}")

def test(dataloader,classifier,filename):
  MAE_model = prepare_mae_model(model_type="gan")
  MAE_model.load_state_dict(torch.load(filename))
  MAE_model.eval()
  for param in MAE_model.parameters():
    param.requires_grad = False
  MAE_model.to(device)
  for param in classifier.parameters():
    param.requires_grad = False
  classifier.to(device)
  classifier.eval()
  step = 0; suc = 0; total = 0
  for x,label in dataloader:
    x = x.to(device); label = label.to(device)
    encoded_token,_,ids_restore= MAE_model.forward_encoder(x,0)
    decoded_token = MAE_model.forward_decoder(encoded_token,ids_restore)
    adv_x = MAE_model.unpatchify(decoded_token)
    adv_x = project(x,adv_x)
    pred_adv = classifier.forward(adv_x)
    suc += (pred_adv.argmax(dim=1)!=label).sum()
    total += x.shape[0]
    step += 1
    if step>=10:
      print(f"suc_rate:{suc/total}")
      break

def test_CDTA(filename,MAE_model=None):
    names = ["birds_400","comic_books","food_101","oxford_102_flower"]
    model_type = "vgg16_bn"
    paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/valid","../../../CDTA_datasets/comic_books/test","../../../CDTA_datasets/food_101/food-101/images","../../../CDTA_datasets/oxford_flower_102/dataset/valid"]
    ckpt_paths = [f"../../../CDTA_models/BIRDS-400/{model_type}.pth.tar",f"../../../CDTA_models/Comic Books/{model_type}.pth.tar",
                  f"../../../CDTA_models/Food-101/{model_type}.pth.tar",f"../../../CDTA_models/Oxford 102 Flower/{model_type}.pth.tar"]
    if MAE_model is None:  
      MAE_model = prepare_mae_model(model_type="gan")
      MAE_model.load_state_dict(torch.load(filename))
    MAE_model.eval()
    MAE_model.to(device)
    suc_rate = []
    preprocess = get_preprocess()
    for param in MAE_model.parameters():
      param.requires_grad = False

    for i,name in enumerate(names):  
      dataset = datasets.ImageFolder(paths[i],transform = preprocess)
      dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=3)
      
      num_classes = len(os.listdir(paths[i]))
      classifier = torchvision.models.vgg16_bn(num_classes=num_classes)
      classifier.eval()
      classifier.load_state_dict(torch.load(ckpt_paths[i]))
      classifier.to(device)
      for param in classifier.parameters():
        param.requires_grad = False
      
      fool = 0; total = 0; step = 0;
      for x,label in dataloader:
        x = x.to(device); label = label.to(device)
        encoded_token,_,ids_restore= MAE_model.forward_encoder(x,0)
        decoded_token = MAE_model.forward_decoder(encoded_token,ids_restore)
        adv_x = MAE_model.unpatchify(decoded_token)
        adv_x = project(x,adv_x)
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
    netG = prepare_mae_model(model_type="gan")
    netG.load_state_dict(torch.load(filename))

    suc_rate = []
    preprocess = get_preprocess()
    netG.to(device)
    netG.eval()

    net = torchvision.models.vgg16(pretrained=True)
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
      _,decoded_token,_ = netG(x,mask_ratio=0)
      adv_x = netG.unpatchify(decoded_token)
      adv_x = project(x,adv_x)
      pred_adv = net(adv_x)
      pred_clean = net(x)
      fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
      total += (pred_clean.argmax(dim=1)==label).sum()
      step += 1
      if step>=20:
        print(f"fool_rate:{fool/total}")
        suc_rate.append((fool/total).cpu().detach().numpy())
        break
      

if __name__ == "__main__":
    preprocess = get_preprocess()
    # classifier = torchvision.models.vgg16(pretrained=True)
    # dataset = datasets.ImageFolder("../../../imagenet-mini/train",transform=preprocess)
    # dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=2,prefetch_factor=4)
    
    # vgg16 = torchvision.models.vgg16(pretrained=True)
    # vgg16_timm = timm.create_model("vgg16",pretrained=True)
    # vgg16_timm.to(device)
    # inc_res = timm.create_model("inception_resnet_v2",pretrained=True)
    
  
    # classifiers[0].load_state_dict(torch.load("../2000.pth"))
    # classifiers[1].load_state_dict(torch.load("../4000.pth"))
    # classifiers[2].load_state_dict(torch.load("../6000.pth"))
    # classifiers[3].load_state_dict(torch.load("../8000.pth"))

    # classifiers.append(torchvision.models.inception_v3(pretrained=True))
    # classifiers.append(torchvision.models.resnet152(pretrained=True)) 
    # classifiers.append(timm.create_model("inception_resnet_v2",pretrained=True))
    # resnet50 = torchvision.models.resnet50(pretrained=True)
    # resnet50.load_state_dict(torch.load("../0_10000.pth"))
    # resnet50.to(device)

    # F_args.append(FIA_args(classifiers[0],"layer2"))
    # F_args.append(FIA_args(classifiers[1],"layer2"))
    # F_args.append(FIA_args(classifiers[2],"layer2"))
    # F_args.append(FIA_args(classifiers[3],"layer2"))
    
    
    train(dataloader,1,10,net,MAE_model=None,filename=filename1,
                      freeze_encoder=True,F_args=None)
    
    






    
