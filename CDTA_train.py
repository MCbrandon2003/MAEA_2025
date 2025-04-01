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
from dataset_utils import get_preprocess,data_augment
import matplotlib.pyplot as plt
import os
from model_utils import MAE_Classifier
from attack_utils import FIA_map,input_diversity,TI_backward_hook
from ensemble_training import train_FIA_ensemble
import timm
from train import project,Norm,Inverse_Norm
from model_utils import VGG16
from CDTA.utils.cdta import CDTAttack
from CDTA.utils.encoder import AttackEncoder
from knowledge_distillation import distillation_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_Transformer(dataloader,epochs,steps_per_epoch,MAE_filename,classifier
                      ):
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    MAE_model = prepare_mae_model(model_type="gan")
    if os.path.exists(MAE_filename):
      MAE_model.load_state_dict(torch.load(MAE_filename))
      print(f"loading from {MAE_filename}")
    MAE_model.to(device)
    optimizer = torch.optim.Adam(MAE_model.parameters(),lr=1e-5) 
    preprocess = get_preprocess()
    for blk in MAE_model.blocks:  
      for param in blk.parameters():
        param.requires_grad = False
    MAE_model.train()
    for epoch in range(epochs):
      step = 0;
      for x,label in dataloader:
        optimizer.zero_grad()
        x = x.to(device); label = label.to(device)
        enc = AttackEncoder("CDTA/pretrained/surrogate/simsiam_bs256_100ep_cst.tar")
        enc.eval()
        enc = enc.to(device)
        Attack = CDTAttack(
            enc, 
            eps=16./255., 
            nb_iter=30, 
            alpha=4./255.
        )
        x_1 = Inverse_Norm(x)
        T_adv_x = Attack(x_1)
        T_adv_x = project(x,Norm(T_adv_x))
        # Fmap = FIA_map(x,label,classifier,"features",30,0.3)
        Fmap = 0
        _,decoded_token,_ = MAE_model(x)
        adv_x = MAE_model.unpatchify(decoded_token)
        adv_x = project(x,adv_x)
        feature_layer = classifier._modules.get("features")
        feature_layer.register_forward_hook(get_mid_output)
        pred_adv = classifier(adv_x)
        T_pred_adv = classifier(T_adv_x)
        loss = distillation_loss(pred_adv,T_pred_adv,label,alpha=1)
        loss.backward()
        optimizer.step()  
        suc_rate = (pred_adv.argmax(dim=1)!=label).sum()/x.shape[0]
        print(f"epoch {epoch} step {step} loss:{loss:.3f} suc_rate:{suc_rate:.3f}")
        step += 1
        if step %30 == 0:
          torch.save(MAE_model.state_dict(),MAE_filename)
          print(f"MAE_model saved to {MAE_filename}")
          

def test_CDTA():
    names = ["birds_400","comic_books","food_101","oxford_102_flower"]
    paths = ["../../../CDTA_datasets/birds_400/BIRDS-400/valid","../../../CDTA_datasets/comic_books/test","../../../CDTA_datasets/food_101/food-101/images","../../../CDTA_datasets/oxford_flower_102/dataset/valid"]
    ckpt_paths = ["../../../CDTA_models/BIRDS-400/inception_v3.pth.tar","../../../CDTA_models/Comic Books/inception_v3.pth.tar",
                  "../../../CDTA_models/Food-101/inception_v3.pth.tar","../../../CDTA_models/Oxford 102 Flower/inception_v3.pth.tar"]
    
    suc_rate = []
    preprocess = get_preprocess()

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
        enc = AttackEncoder("CDTA/pretrained/surrogate/simsiam_bs256_100ep_cst.tar")
        enc.eval()
        enc = enc.to(device)
        Attack = CDTAttack(
            enc, 
            eps=16./255., 
            nb_iter=30, 
            alpha=4./255.
        )
        x_1 = Inverse_Norm(x).float()
        adv_x = Attack(x_1)
        adv_x = project(x,Norm(adv_x))
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

def test_CDTA_ImageNet():
    suc_rate = []
    preprocess = get_preprocess()

    dataset = datasets.ImageFolder("../../../imagenet-mini/val",transform = preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=3)

    classifier = torchvision.models.inception_v3(pretrained=True)
    classifier.eval()
    classifier.to(device)
    for param in classifier.parameters():
      param.requires_grad = False
    
    fool = 0; total = 0; step = 0;
    for x,label in dataloader:
      x = x.to(device); label = label.to(device)
      enc = AttackEncoder("CDTA/pretrained/surrogate/simsiam_bs256_100ep_cst.tar")
      enc.eval()
      enc = enc.to(device)
      Attack = CDTAttack(
          enc, 
          eps=16./255., 
          nb_iter=30, 
          alpha=4./255.
      )
      x_1 = Inverse_Norm(x).float()
      adv_x = Attack(x_1)
      adv_x = project(x,Norm(adv_x))
      pred_adv = classifier.forward(adv_x)
      pred_clean = classifier.forward(x)
      fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
      total += (pred_clean.argmax(dim=1)==label).sum()
      step += 1
      if step>=10:
        print(f"fool_rate:{fool/total}")
        suc_rate.append((fool/total).cpu().detach().numpy())
        break
    print(np.mean(suc_rate))

def test_CDTA_B7():
    suc_rate = []
    preprocess = get_preprocess()

    dataset = datasets.ImageFolder("../../../CDTA_datasets/food_101/food-101/images",
                                    transform = preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=3)

    # 加载预训练的 EfficientNet-B7 模型
    model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
    num_classes = 101
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load("SAM_models/EfficientNetB7_food101"))
    model.to(device)
    model.eval()
    for param in model.parameters():
      param.requires_grad = False
    
    fool = 0; total = 0; step = 0; Sum = 0
    for x,label in dataloader:
      x = x.to(device); label = label.to(device)
      enc = AttackEncoder("CDTA/pretrained/surrogate/simsiam_bs256_100ep_cst.tar")
      enc.eval()
      enc = enc.to(device)
      Attack = CDTAttack(
          enc, 
          eps=16./255., 
          nb_iter=30, 
          alpha=4./255.
      )
      x_1 = Inverse_Norm(x).float()
      adv_x = Attack(x_1)
      adv_x = project(x,Norm(adv_x))
      pred_adv = model.forward(adv_x)
      pred_clean = model.forward(x)
      fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
      total += (pred_clean.argmax(dim=1)==label).sum()
      step += 1
      Sum += x.shape[0]
      if step>=10:
        print(f"fool_rate:{fool/total},acc:{total/Sum}")
        suc_rate.append((fool/total).cpu().detach().numpy())
        break
    print(np.mean(suc_rate))

if __name__ == '__main__':
    preprocess = get_preprocess()
    dataset = torchvision.datasets.ImageFolder("../../../imagenet-mini/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=32,\
                                                num_workers=2,prefetch_factor=4)
    inception_v3 = timm.create_model("inception_v3",pretrained=True)
    inception_v3.eval()
    inception_v3.to(device)
    vgg16 = timm.create_model("vgg16",pretrained=True)
    vgg16.eval()
    vgg16.to(device)
    train_Transformer(dataloader,epochs=5,steps_per_epoch=30,classifier=vgg16,
                            MAE_filename="mae_models/CMAE_v3.pth")
    test_CDTA()



