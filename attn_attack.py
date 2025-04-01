from timm.models.xception_aligned import xception41
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
from ensemble_training import train_FIA_ensemble
import timm
from model_utils import VGG16
from torchvision import transforms
from sam.example.utility.bypass_bn import enable_running_stats,disable_running_stats
from sam.example.model.smooth_cross_entropy import smooth_crossentropy
from sam.sam import SAM
from CDA.gaussian_smoothing import get_gaussian_kernel
from gradcam import generate_grad_cam,top_30_percent_gradcam
from ADA.gradcam import GradCAM
from attack_utils import project
from train import test_CDTA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        
def train(dataloader,epochs,steps_per_epoch,classifier,MAE_model=None,filename="Generator_trained_vanilla.pth",
          mask_ratio=0.5,freeze_encoder=False,lda=1):
    classifier.eval()
    classifier.to(device)
    for param in classifier.parameters():
      param.requires_grad = False

    if MAE_model is None:
        MAE_model = prepare_mae_model(model_type="gan")
    MAE_model.shuffle = False
    MAE_model.to(device)
    optimizer = torch.optim.Adam(MAE_model.parameters(),lr=1e-5) 

    if freeze_encoder == True:
        for blk in MAE_model.blocks:
            for param in blk.parameters():
                param.requries_grad = False
    
    if os.path.exists(filename):
        print(f"loading from {filename}")
        MAE_model.load_state_dict(torch.load(filename))
    
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o

    criterion_gcam = nn.MSELoss(reduction='mean')
    for epoch in range(epochs):
        step = 0
        for x,label in dataloader:
            optimizer.zero_grad()
            x = x.to(device);label = label.to(device)        
            _,decoded_token,_ = MAE_model(x,0)
            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x, adv_x)
            x.requires_grad_(); adv_x.requires_grad_()
            adv_output = classifier(adv_x)
            output = classifier(x)
            # attn_loss = 0

            # ori_gcam = GradCAM(model=classifier,candidate_layers=["features.30"])
            # index, output = ori_gcam.forward(x)
            # ori_gcam.backward(ids=index,output=output)
            # ori_attn = ori_gcam.generate(target_layer="features.30")

            # adv_gcam = GradCAM(model=classifier,candidate_layers=["features.30"])
            # _,adv_output = adv_gcam.forward(adv_x)
            # adv_gcam.backward(ids=index,output=adv_output)
            # adv_attn = adv_gcam.generate(target_layer="features.30")
            
            # ce_loss = -F.cross_entropy(adv_output - output,label)

            # for a_attn,o_attn in zip(adv_attn,ori_attn):
            #   attn_loss += criterion_gcam(a_attn,o_attn)
            
            loss = -F.cross_entropy(adv_output - output,label)
            loss.backward()
            optimizer.step()

            # ori_gcam.remove_hook()
            # ori_gcam.clear()
            # adv_gcam.remove_hook()
            # adv_gcam.clear()

            suc_rate = (adv_output.argmax(dim=1)!=label).sum() / x.shape[0]
            print(f"epoch {epoch} step {step}:\
CE_loss:{loss:.2f} ,suc_rate:{suc_rate}")
            step += 1
            if(step>=steps_per_epoch):
                break
        torch.save(MAE_model.state_dict(),filename)
        print(f"Model saved to {filename}")
        test_CDTA(filename)

if __name__ == "__main__":
    preprocess = get_preprocess()
    classifier = torchvision.models.vgg16(pretrained=True)
    dataset = datasets.ImageFolder("../../../imagenet-mini/train",transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=2,prefetch_factor=4)
    
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_timm = timm.create_model("vgg16",pretrained=True)



    filename1 = "mae_models/vgg16_attn.pth"
    train(dataloader,30,20,vgg16_timm,MAE_model=None,filename=filename1,
          mask_ratio=0.3,freeze_encoder=True,lda=0)
    

