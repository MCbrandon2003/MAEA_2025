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
def get_CDTA_dataloader(dataset):
    dataset_paths = {"birds-400": "CDTA/dataset/birds-400/valid",
                     "comic-books": "CDTA/dataset/comic-books/test",
                     "food-101": "CDTA/dataset/food-101/valid", 
                     "oxford-102-flower": "CDTA/dataset/oxford-102-flower/valid"
                    }
    preprocess = get_preprocess()
    ds = datasets.ImageFolder(dataset_paths[dataset],transform = preprocess)
    dataloader = torch.utils.data.DataLoader(ds,batch_size=32,shuffle=True,num_workers=4,prefetch_factor=3)
    return dataloader

def get_CDTA_models(dataset, model_name):
    
    path = f"CDTA/pretrained/target/{dataset}/{model_name}.pth.tar"
    
    if dataset == 'comic-books':
        num_classes = 86
    elif dataset == 'oxford-102-flower':
        num_classes = 102
    elif dataset == 'birds-400':
        num_classes = 400
    elif dataset == 'food-101':
        num_classes = 101
        
    if model_name.startswith('inception'):
        net = torchvision.models.__dict__['inception_v3'](aux_logits=False, init_weights=True, num_classes=num_classes)
    else:
        net = torchvision.models.__dict__[model_name](num_classes=num_classes)
        
    net.load_state_dict(torch.load(path))
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    net = net.to(device)
    return net
        

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


def write_log(message, filename="logfile.log"):
    """
    将给定的消息写入指定的日志文件。

    参数:
    message (str): 要写入日志文件的消息字符串。
    filename (str): 日志文件的名称，默认为 'logfile.log'。
    """
    try:
        # 使用 'a' 模式来追加内容，如果文件不存在，将会创建新文件
        with open(filename, 'a') as file:
            file.write(message + '\n')  # 添加换行符以便每条消息单独一行
    except Exception as e:
        print(f"Error writing to log file: {e}")


def test_MAE_CDTA(filename, info=""):
    datasets = ["birds-400","comic-books","food-101","oxford-102-flower"]
    models = ["densenet161","inception_v3","resnet34","vgg16_bn"]
    write_log(f"info={info}")
    
    MAE_model = prepare_mae_model(model_type="gan")
    MAE_model.shuffle = False
    MAE_model.to(device)
    MAE_model.eval()

    if os.path.exists(filename):
        print(f"loading from {filename}")
        MAE_model.load_state_dict(torch.load(filename))

    for param in MAE_model.parameters():
        param.requires_grad = False
        
    for dataset in datasets:
        dataloader = get_CDTA_dataloader(dataset)
        print(f"start {dataset}")
        for model in models:
          net = get_CDTA_models(dataset, model)
          fool = 0; total = 0; step = 0;
            
          for x, label in dataloader:
            x = x.to(device); label = label.to(device)
            encoded_token,_,ids_restore= MAE_model.forward_encoder(x,0)
            decoded_token = MAE_model.forward_decoder(encoded_token,ids_restore)
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x,adv_x)
            pred_adv = net.forward(adv_x)
            pred_clean = net.forward(x)
            fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
            total += (pred_clean.argmax(dim=1)==label).sum()
            step += 1
            if step>=20:
              print(f"{dataset} {model} fool_rate:{fool/total}")
              write_log(f"{dataset} {model} fool_rate:{fool/total}")
              break

def train(dataloader, epochs, MAE_model=None, filename="MAE_CDTA_KD.pth",
          mask_ratio=None, freeze_encoder=True):

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

    for epoch in range(epochs):
        step = 0
        for x,label in dataloader:
            optimizer.zero_grad()
            x = x.to(device);label = label.to(device)
            _,decoded_token,_ = MAE_model(x,mask_ratio=0)
            optimizer.zero_grad()
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x,adv_x)

            # 计算CDTA的损失函数
            CDTA_x = Attack.perturb(Inverse_Norm(x)).detach()
            loss = Attack.calculate_loss(adv_x, Norm(CDTA_x))
            loss.backward()
            optimizer.step()
            print(f"epoch {epoch} step {step}: loss:{loss} ")
            step += 1
            if(step % 30 == 0):
              torch.save(MAE_model.state_dict(),filename)
              print(f"Model saved to {filename}")

if __name__ == '__main__':
    preprocess = get_preprocess()
    dataset = torchvision.datasets.ImageFolder("CDTA/dataset/imagenet-mini/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=32,\
                                                num_workers=2,prefetch_factor=4)
    train(dataloader, 2, filename = "MAE_CDTA.pth")
    # test_MAE_CDTA("MAE_CDTA.pth","MAE_CDTA")


