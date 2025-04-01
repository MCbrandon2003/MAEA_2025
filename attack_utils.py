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
  return (x*imagenet_std + imagenet_mean).float()

def Norm(x):
  imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1))\
                  .float().to(device)
  imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1))\
                  .float().to(device)
  return ((x-imagenet_mean)/imagenet_std).float()

def project(x0,adv_x,eps=16./255.):
    x0 = Inverse_Norm(x0)
    adv_x = Inverse_Norm(adv_x)
    pert = torch.clamp((adv_x-x0),-eps,eps)
    adv_x = torch.clamp((pert+x0),0,1)
    return Norm(adv_x)

def input_diversity(input_tensor, image_size=224, image_resize=250, prob=0.7):
    """
    Input diversity: https://arxiv.org/abs/1803.06978
    PyTorch version
    """
    random_prob = torch.rand(1).item()
    if random_prob > prob:
      return input_tensor
    
    batch_size, _, _, _ = input_tensor.size()
    rnd = torch.randint(low=image_size, high=image_resize, size=(1,)).item()
    rescaled = F.interpolate(input_tensor, size=(rnd, rnd), mode='nearest')
    
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    
    pad_top = torch.randint(low=0, high=h_rem, size=(1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem, size=(1,)).item()
    pad_right = w_rem - pad_left
    
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded = F.pad(rescaled, padding, value=0)
    
    padded = padded.view(batch_size, 3, image_resize, image_resize)

    ret = F.interpolate(padded, size=(image_size, image_size), mode='nearest')
    
    return ret

def Gaussian_kernel(k=15):
    kernel = np.zeros(shape=(k,k))
    sig = k/np.sqrt(3)
    for i in range(k):
      for j in range(k):
        d_i = np.abs(i-7)
        d_j = np.abs(j-7)
        kernel[i][j] = 1/(2*np.pi*np.square(sig)) * np.exp(-(np.square(d_i)+np.square(d_j)) / (2*np.square(sig)))
    kernel = np.zeros(shape=(3,k,k)) + kernel.reshape(1,*kernel.shape)
    return torch.from_numpy(kernel).unsqueeze(1).float()

def gaussian_noise(x):
  return x + torch.from_numpy(np.random.normal(0,0.04,size=x.shape)).to(device)

block_rate = 0.1

def random_block(x):
  return x * torch.from_numpy(np.random.choice(a=[0,1],p=[block_rate,1-block_rate],size=x.shape)).to(device)

def get_lr(epoch):
    epochs=10
    decay=0.1
    initial_lr = 3.735
    n=0.85
    total = 0
    lr = initial_lr*np.power(n,epoch)
    return lr
class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).to(device)
        std = torch.as_tensor(std).to(device)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class Momentum:
  def __init__(self,u):
    self.u = u
    self.momentum = 0
  def call(self,grad):
    u = self.u
    norm = torch.norm(grad.reshape(grad.shape[0],-1),p=1,dim=1).reshape(grad.shape[0],1,1,1)
    self.momentum = u*self.momentum + grad/norm
    return self.momentum



invNormalize = NormalizeInverse([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
Normlize_Trans = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def update_adv(X, X_pert, pert, epsilon):
    X = X.clone().detach()
    X_pert = X_pert.clone().detach()
    X_pert = invNormalize(X_pert)
    X = invNormalize(X)
    X_pert = X_pert + pert
    noise = (X_pert - X).clamp(-epsilon, epsilon)
    X_pert = X + noise
    X_pert = X_pert.clamp(0, 1)
    X_pert = Normlize_Trans(X_pert)
    return X_pert.clone().detach()


def basic_mae_attack(x,mae_model,mask_ratio,steps,factor,eps,random_aug=None,trick=None,u=1.0):
  adv_x = x.detach().float()
  x_orig = x.clone().detach().float()
  patch_x_orig = mae_model.patchify(x_orig)
 
  M = Momentum(u)

  for step in range(steps):
    if(random_aug==None):
      x1 = adv_x
    else:
      x1 = random_aug(adv_x).detach().float()

    if(trick=="Nesterov"):
      x1 = adv_x.detach() + u*factor*M.momentum
    x1.requires_grad_()
    
    latent, mask, ids_restore = mae_model.forward_encoder(x1, mask_ratio)
    pred = mae_model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    loss = torch.square(pred-patch_x_orig).mean()
    
    print(f"at step:{step+1} , loss={loss}")
    loss.backward()

    if(trick=="Momentum" or trick=="Nesterov"):
      grad = M.call(x1.grad)
    elif(trick=="Translation_Invariant"):
      kernel = Gaussian_kernel().to(device)
      grad = M.call(x1.grad)
      grad = F.conv2d(grad,kernel,padding="same",groups=3)
      
    else:
      grad = x1.grad
    
    pert = factor * torch.sign(grad)
    adv_x = update_adv(x,adv_x,pert,eps)
  return adv_x

def BIM(x,labels,source_model,eps,factor,steps,trick=None,u=1.0):
  adv_x = x.detach().float()
  L = torch.nn.CrossEntropyLoss()
  if(trick=="Momentum" or trick=="Nesterov"):
    M = Momentum(u)

  for step in range(steps):
    adv_x.requires_grad_()
    # _,y,_1 = mae_model(adv_x,mask_ratio)
    # y = mae_model.unpatchify(y)
    if(trick != "Nesterov"): 
      pred = source_model(adv_x)
      loss = L(pred,labels)
      print(f"step{step},loss={loss}")
      loss.backward()
    if(trick=="Momentum"):  
      grad = M.call(adv_x.grad)
    elif(trick=="Nesterov"):
      x_1 = adv_x.detach() + u*factor*M.momentum
      x_1.requires_grad_()
      pred = source_model(x_1)
      loss = L(pred,labels)
      print(f"step{step},loss={loss}")
      loss.backward()
      grad = M.call(x_1.grad)
    elif(trick=="Translation_Invariant"):
      kernel = Gaussian_kernel().to(device)
      grad = F.conv2d(adv_x.grad,kernel,padding="same",groups=3)
      
    else:
      grad = adv_x.grad
   
    pert = factor * torch.sign(grad)
    adv_x = update_adv(x,adv_x,pert,eps)
  return adv_x


mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
std_np = std.permute(1,2,0).numpy()
mean_np = mean.permute(1,2,0).numpy()
def deprocess(x):
  X = x.cpu().permute(1,2,0).numpy()
  X = X*std_np+mean_np
  X = np.clip(X,0,1)
  return X

def basic_mae_attack_experiment(dataloader,data_batch_size,data_steps,
                 mae_model,mask_ratio,steps,factor,eps,target_model,random_aug=None,trick=None,u=1):
  mae_model.to(device)
  data_step = 0
  adv_x_buffer = torch.zeros(size=(data_batch_size*data_steps,3,224,224))
  acc = 0
  for x,y in dataloader:
    if(data_step>=data_steps):
      break
    print("step:",data_step+1)
    data_step += 1
    x = x.to(device);y = y.to(device)
    adv_x = basic_mae_attack(x,mae_model,mask_ratio,steps,factor,eps,random_aug,trick,u)
    #adv_x_buffer[(data_step-1)*data_batch_size:data_step*data_batch_size,:,:,:] = adv_x
    pred = target_model(adv_x).argmax(dim=1)
    acc += (pred==y).sum() 
  #torch.save(adv_x_buffer.cpu(),"adv_x.pt")
  print("acc:",acc/(data_steps*data_batch_size))
  return acc/(data_steps*data_batch_size)

def BIM_experiment(dataloader,data_batch_size,data_steps,source_model,
          target_model,steps,factor,eps,trick=None,u=1):
  source_model.to(device);target_model.to(device);mae_model.to(device)
  data_step = 0
  acc = 0
  adv_x_buffer = torch.zeros(size=(data_batch_size*data_steps,3,224,224))
  for x,labels in dataloader:
    x = x.to(device);labels = labels.to(device)
    adv_x = BIM(x,labels,source_model,eps,factor,steps,trick,u)
    data_step += 1
    adv_x_buffer[(data_step-1)*data_batch_size:data_step*data_batch_size,:,:,:] = adv_x
    print(f"data_step:{data_step}")
    pred = target_model(adv_x).argmax(dim=1)
    acc += (pred==labels).sum()
    if(data_step>=data_steps):
      break
  torch.save(adv_x_buffer,"BIM_x.pt")
  print("acc:",acc/(data_steps*data_batch_size))
  return acc/(data_steps*data_batch_size)


def FIA_map(X,y,model,layer_name,N=30,drop_rate=0.3):

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
        X_random = torch.zeros(X.size()).cuda()
        X_random.copy_(X).detach()
        X_random.requires_grad = True
        Mask = torch.bernoulli(torch.ones_like(X_random)*(1-drop_rate))
        X_random = X_random * Mask
        output_random = model(X_random)        
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

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

def TI_gradient(grad):
    kernel = gkern(15, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    stack_kernel = torch.from_numpy(stack_kernel).permute(2,3,0,1).to(device)
    #print(stack_kernel.shape)
    grad = torch.nn.functional.conv2d(grad, stack_kernel, stride=1, padding="same",groups = 3)
    grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
    return grad

def FIA_attack(x,label ,model,target_model,layer_name):
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    model.to(device); 
    model.eval(); 
    # target_model.to(device)
    # target_model.eval()

    momentum = 0
    fool = 0; total = 0
    x = x.to(device)
    label = label.to(device)
    Map = FIA_map(x,label,model,layer_name)
    model._modules.get(layer_name).register_forward_hook(get_mid_output)
    adv_x = x.clone()
    for i in range(10):         
      adv_x.requires_grad = True
      model.forward(adv_x)   
      loss = (Map*mid_output).sum()
      loss.backward()
      grad = adv_x.grad
      a = torch.sum(torch.abs(grad),dim=[1,2,3])
      with torch.no_grad():
        momentum =  momentum + grad / torch.sum(torch.abs(grad),dim=[1,2,3],keepdim=True)
        adv_x = adv_x - 0.274 * torch.sign(momentum)/10
        adv_x = project(x,adv_x)
    return x

def get_feature_and_logit_FIA(x, model, layer_name):
  def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
  model.to(device); 
  model.eval();
  model._modules.get(layer_name).register_forward_hook(get_mid_output)
  logit = model.forward(x)
  return mid_output, logit

def get_feature_and_logit_CDTA(x, classifier, cdtattack):
  classifier.to(device)
  classifier.eval()
  logit = classifier.forward(x)
  vec = cdtattack.enc(x)
  features = [cdtattack.enc.net.feature1,
              cdtattack.enc.net.feature2,
              cdtattack.enc.net.feature3,
              vec]
  return logit, vec

def TI_attack(dataloader,model):
  model.eval()
  correct = 0; total = 0 ;step = 0
  print("start_validation")
  for x,label in dataloader:
      x = x.to(device)
      label = label.to(device)
      adv_x = x.clone()
      pred = model(x)
      total += x.shape[0]
      correct += (pred.argmax(dim=1)==label).sum()
      step += 1
      if step >= 10:
        break
  print(f"acc:{correct/total}")
        

def TI_backward_hook(module, grad_input, grad_output):
   new_grad_input = tuple(TI_gradient(gi) if gi.shape[-1]==224 else gi for gi in grad_input)
   return new_grad_input

if __name__=='__main__':
    preprocess = get_preprocess()
    models = []
    vgg16 = timm.create_model("vgg16",pretrained=True)
    inception_v3 = torchvision.models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
    
  
    dataset = torchvision.datasets.ImageFolder("../../../imagenet-mini/val",preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=50,shuffle=True,\
                                              num_workers=2)
    FIA_attack(dataloader,inception_v3,vgg16,"Mixed_5b")
  






