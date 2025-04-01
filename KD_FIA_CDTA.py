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
from attack_utils import FIA_map,input_diversity,TI_backward_hook,get_feature_and_logit_FIA
from ensemble_training import train_FIA_ensemble
import timm
from model_utils import VGG16
from torchvision import transforms
from attack_utils import FIA_attack
from knowledge_distillation import distillation_loss
from train import test_CDTA
from NAA import NAA_attack
from CDTA.utils.cdta import CDTAttack
from CDTA.utils.encoder import AttackEncoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
  return x*imagenet_std + imagenet_mean

def Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).to(device)
  return (x-imagenet_mean)/imagenet_std

def project(x0,adv_x,eps=16./255.):
    x0 = Inverse_Norm(x0)
    adv_x = Inverse_Norm(adv_x)
    pert = torch.clamp((adv_x-x0),-eps,eps)
    adv_x = torch.clamp(pert + x0,0,1)
    return Norm(adv_x).float()
        
def train(dataloader,epochs,classifier,MAE_model=None,filename="Generator_trained_vanilla.pth",
          mask_ratio=None,freeze_encoder=False,F_args=None, alpha = 1, beta = 1):
    classifier.eval()
    classifier.to(device)

    enc = AttackEncoder("CDTA/pretrained/surrogate/simsiam_bs256_100ep_cst.tar")
    enc.eval()
    enc = enc.to(device)
    Attack = CDTAttack(
        enc, 
        eps=16./255., 
        nb_iter=30, 
        alpha=4./255.
    )
    
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
            optimizer.zero_grad()
            x = x.to(device);label = label.to(device)
            _,decoded_token,_ = MAE_model(x,mask_ratio=0)
            FIA_x = FIA_attack(x,label,F_args.model,None,F_args.layer_name)
            optimizer.zero_grad()
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x,adv_x)
            
            # 计算FIA的损失函数
            feature_student, y_pred_student = get_feature_and_logit_FIA(adv_x, classifier, "features")
            feature_FIA, y_pred_FIA = get_feature_and_logit_FIA(FIA_x, classifier, "features")
            pred_loss_FIA = distillation_loss(y_pred_student,y_pred_FIA,label,alpha=1)
            feature_loss_FIA = torch.norm(feature_student - feature_FIA) / \
                              torch.norm(feature_student)

            # 计算CDTA的损失函数
            y_pred_student = classifier(adv_x)
            CDTA_x = Attack.perturb(x)
            y_pred_CDTA = classifier(CDTA_x)
            feature_loss_CDTA = Attack.calculate_loss(adv_x, CDTA_x)
            pred_loss_CDTA = distillation_loss(y_pred_CDTA, feature_loss_CDTA, label,
                              alpha = 1)
            
            loss = (feature_loss_CDTA + beta * pred_loss_CDTA) + \
                    alpha * (feature_loss_FIA + beta * pred_loss_FIA)


            loss.backward()
            optimizer.step()
            suc_rate = (y_pred_student.argmax(dim=1)!=label).sum() / x.shape[0]
            print(f"epoch {epoch} step {step}: loss:{loss} suc_rate:{suc_rate}")
            step += 1
            if(step%30==0):
              torch.save(MAE_model.state_dict(),filename)
              print(f"Model saved to {filename}")

if __name__ == "__main__":
    preprocess = get_preprocess()
    classifier = torchvision.models.vgg16(pretrained=True)
    dataset = datasets.ImageFolder("../../../imagenet-mini/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,drop_last=True,
                  shuffle=True,num_workers=2,prefetch_factor=4)
    vgg16_timm = timm.create_model("vgg16",pretrained=True)
    F_args = FIA_args(vgg16_timm,"features")
    filename1 = "mae_models/vgg16_timm_NAA_KL_0.pth"
    train(dataloader,10,vgg16_timm,F_args=F_args)

    






