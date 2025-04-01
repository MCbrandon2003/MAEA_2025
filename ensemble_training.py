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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def Inverse_Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).to(device)
  return x*imagenet_std + imagenet_mean

def Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).to(device)
  return (x-imagenet_mean)/imagenet_std

def project(x0,adv_x,eps=16./255.):
    x0 = Inverse_Norm(x0)
    adv_x = Inverse_Norm(adv_x)
    pert = torch.clamp((adv_x-x0),-eps,eps)
    return Norm(pert + x0).float()
        
def train(dataloader,epochs,steps_per_epoch,classifiers,MAE_model=None,filename="ensemble_G.pth",
          mask_ratio=None,freeze_encoder=False):
    for classifier in classifiers:
      classifier.eval()
      classifier.to(device)
    if MAE_model is None:
        MAE_model = prepare_mae_model(model_type="gan")
    
    MAE_model.shuffle = False
    MAE_model.to(device)
    
    if freeze_encoder == True:
        for blk in MAE_model.blocks:
            for param in blk.parameters():
                param.requries_grad = False
    
    if mask_ratio is None:  
      MAE_model.shuffle = False
      mask_ratio = 0
    else:
      MAE_model.shuffle = True

    optimizer = torch.optim.Adam(MAE_model.parameters(),lr=1e-5)

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
            x = x.to(device);label = label.to(device)
            encoded_token,_,ids_restore= MAE_model.forward_encoder(x,mask_ratio)
            decoded_token = MAE_model.forward_decoder(encoded_token,ids_restore)
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x,adv_x)
            
            loss = 0
            for classifier in classifiers:
              pred_adv = classifier.forward(adv_x)
              pred_clean = classifier.forward(x)
              loss += -F.cross_entropy(pred_adv - pred_clean,label)

            loss.backward()
            optimizer.step()
            suc_rate = (pred_adv.argmax(dim=1)!=label).sum() / x.shape[0]
            print(f"epoch {epoch} step {step}: loss:{-loss} suc_rate:{suc_rate}")
            step += 1
            if(step>=steps_per_epoch):
                break
        torch.save(MAE_model.state_dict(),filename)
        print(f"Model saved to {filename}")

def train_FIA_ensemble(dataloader,epochs,steps_per_epoch,classifiers,MAE_model=None,filename="Generator_trained_vanilla.pth",
          mask_ratio=None,freeze_encoder=False,F_args=None,DI=False):
    for classifier in classifiers:  
      classifier.eval()
      classifier.to(device)
      for param in classifier.parameters():
        param.requires_grad = False
    if MAE_model is None:
        MAE_model = prepare_mae_model(model_type="gan")
    MAE_model.shuffle = False
    MAE_model.to(device)
    
    if freeze_encoder == True:
        for blk in MAE_model.blocks:
            for param in blk.parameters():
                param.requries_grad = False
    
    if mask_ratio is None:  
      MAE_model.shuffle = False
      mask_ratio = 0
    else:
      MAE_model.shuffle = True

    optimizer = torch.optim.Adam(MAE_model.parameters(),lr=1e-5)

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
            loss = 0
            optimizer.zero_grad()
            x = x.to(device);label = label.to(device)
            if DI==True:
              x = input_diversity(x)
            F_map = []
            for classifier,F_arg in zip(classifiers,F_args):
              F_map.append(FIA_map(x,label,F_arg.model,F_arg.layer_name,F_arg.N,F_arg.drop_rate))
            encoded_token,_,ids_restore= MAE_model.forward_encoder(x,mask_ratio)
            decoded_token = MAE_model.forward_decoder(encoded_token,ids_restore)
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x,adv_x)        
            if F_args is None:
              pred_adv = classifier.forward(adv_x)
              pred_clean = classifier.forward(x)
              loss = -F.cross_entropy(pred_adv - pred_clean,label)
            else:
              for i,classifier in enumerate(classifiers):
                if F_args[i].layer_name=="14":
                  feature_layer = classifier._modules.get("features")._modules.get(F_args[i].layer_name)
                else:
                  feature_layer = classifier._modules.get(F_args[i].layer_name)
                feature_layer.register_forward_hook(get_mid_output)
                pred_adv = classifier.forward(adv_x)           
                loss += (F_map[i]*mid_output).sum()/30
            loss.backward()
            optimizer.step()
            suc_rate = (pred_adv.argmax(dim=1)!=label).sum() / x.shape[0]
            print(f"epoch {epoch} step {step}: loss:{-loss} suc_rate:{suc_rate}")
            step += 1
            if(step>=steps_per_epoch):
                break
        torch.save(MAE_model.state_dict(),filename)
        print(f"Model saved to {filename}")
def test_CDTA(filename,MAE_model=None):
    names = ["birds_400","comic_books","food_101","oxford_102_flower"]
    paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/valid","../../../CDTA_datasets/comic_books/test","../../../CDTA_datasets/food_101/food-101/images","../../../CDTA_datasets/oxford_flower_102/dataset/valid"]
    ckpt_paths = ["../../../CDTA_models/BIRDS-400/inception_v3.pth.tar","../../../CDTA_models/Comic Books/inception_v3.pth.tar",
                  "../../../CDTA_models/Food-101/inception_v3.pth.tar","../../../CDTA_models/Oxford 102 Flower/inception_v3.pth.tar"]
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
      classifier = torchvision.models.inception_v3(num_classes=num_classes,aux_logits=False)
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
        if step>=10:
          print(f"{name} fool_rate:{fool/total}")
          suc_rate.append((fool/total).cpu().detach().numpy())
          break
    print(np.mean(suc_rate))
      
if __name__ == "__main__":
    preprocess = get_preprocess()
    classifier = []
    classifier.append(torchvision.models.inception_v3(pretrained=True))
    classifier.append(torchvision.models.resnet50(pretrained=True))
    classifier.append(torchvision.models.vgg16_bn(pretrained=True))
    dataset = datasets.ImageFolder("../../../imagenet_domains/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=4)
    filename = "mae_models/Generator_ensemble_unmasked.pth"
    
    train(dataloader,10,20,classifier,MAE_model=None,filename=filename,
          mask_ratio=None,freeze_encoder=True)
    # train_MD(MD_dataloader,30,20,MCF,classifier,filename=filename)
    
    test_CDTA(filename)

