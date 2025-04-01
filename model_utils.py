import modulefinder
import torch
from mae_pytorch import models_mae
import torchvision
import torchvision.models as models
from torch import nn
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_mae_model(arch='mae_vit_large_patch16',model_type="basic"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    if(model_type=="basic"):
      chkpt_dir = "mae_models/mae_visualize_vit_large.pth"
    elif(model_type=="gan"):
      chkpt_dir = "mae_models/mae_visualize_vit_large_ganloss.pth"
    elif(model_type=="trained"):
      chkpt_dir = "mae_models/mae_generator_trained.pth"
      checkpoint = torch.load(chkpt_dir, map_location='cpu')
      msg = model.load_state_dict(checkpoint, strict=False)
      print(msg)
      return model
    elif(model_type=="trained_v2"):
      chkpt_dir = "mae_models/mae_generator_trained_v2.pth"
      checkpoint = torch.load(chkpt_dir, map_location='cpu')
      msg = model.load_state_dict(checkpoint, strict=False)
      print(msg)
      return model
    elif(model_type=="trained_v3"):
      chkpt_dir = "mae_models/mae_generator_trained_v3.pth"
      checkpoint = torch.load(chkpt_dir, map_location='cpu')
      msg = model.load_state_dict(checkpoint, strict=False)
      print(msg)
      return model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    
    print(msg)
    return model

def get_PACS_classifier(build=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if(build==True):
        model = models.resnet50(num_classes=7)
        torch.save(model.state_dict(),"classifiers/PACS.ckpt")
        model.to(device)
        return model
    else:
        model = models.resnet50(num_classes=7)
        model.load_state_dict(torch.load("classifiers/PACS.ckpt"))
        model.to(device)
        return model

def get_VLCS_classifier(build=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if(build==True):
        model = models.resnet50(num_classes=5)
        torch.save(model.state_dict(),"classifiers/VLCS.ckpt")
        model.to(device)
        return model
    else:
        model = models.resnet50(num_classes=5)
        model.load_state_dict(torch.load("classifiers/VLCS.ckpt"))
        model.to(device)
        return model

class MAE_Classifier(nn.Module):
    def __init__(self,MAE_model,linear_head,path):
        super(MAE_Classifier,self).__init__()
        self.MAE_model = MAE_model
        self.linear_head = linear_head
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(path):
          self.load_state_dict(torch.load(path))
    def decoder_requires_grad_(self,flag=True):
        for blk in self.MAE_model.decoder_blocks:
            blk.requires_grad = flag
    def encoder_requires_grad_(self,flag=True):
        for blk in self.MAE_model.blocks:
            blk.requires_grad = flag
    def forward(self,x):
        encoded_tokens,mask,ids_restore = self.MAE_model.forward_encoder(x,0)
        cls_token = encoded_tokens[:,0,:]
        pred = self.linear_head(cls_token)
        return pred

def prepare_classifier(source_data,module_type):
  birds_400_models = {"densenet161":"CDTA_models/BIRDS-400/densenet161.pth.tar",
             "inception_v3":"CDTA_models/BIRDS-400/inception_v3.pth.tar",
             "resnet34":"CDTA_models/BIRDS-400/resnet34.pth.tar",
             "vgg16_bn":"CDTA_models/BIRDS-400/vgg16_bn.pth.tar"
             }
  comic_books_models = {"densenet161":"CDTA_models/Comic Books/densenet161.pth.tar",
             "inception_v3":"CDTA_models/Comic Books/inception_v3.pth.tar",
             "resnet34":"CDTA_models/Comic Books/resnet34.pth.tar",
             "vgg16_bn":"CDTA_models/Comic Books/vgg16_bn.pth.tar"
             }
  food_101_models = {"densenet161":"CDTA_models/Food-101/densenet161.pth.tar",
             "inception_v3":"CDTA_models/Food-101/inception_v3.pth.tar",
             "resnet34":"CDTA_models/Food-101/resnet34.pth.tar",
             "vgg16_bn":"CDTA_models/Food-101/vgg16_bn.pth.tar"
             }   
  oxford_102_flower = {"densenet161":"CDTA_models/Oxford 102 Flower/densenet161.pth.tar",
             "inception_v3":"CDTA_models/Oxford 102 Flower/inception_v3.pth.tar",
             "resnet34":"CDTA_models/Oxford 102 Flower/resnet34.pth.tar",
             "vgg16_bn":"CDTA_models/Oxford 102 Flower/vgg16_bn.pth.tar"
             }   
  src = {"birds_400":birds_400_models,"comic_books":comic_books_models,
      "food_101":food_101_models,"oxford_102_flower":oxford_102_flower}
  path = src[source_data][module_type]
  dataset = source_data
  if dataset == 'comic_books':
        num_classes = 86
  elif dataset == 'oxford_102_flower':
        num_classes = 102
  elif dataset == 'birds_400':
        num_classes = 400
  elif dataset == 'food_101':
        num_classes = 101
        
  if module_type=='inception_v3':
      net = models.__dict__['inception_v3'](aux_logits=False, init_weights=True, num_classes=num_classes)
  else:
      net = models.__dict__[module_type](num_classes=num_classes)
      
  net.load_state_dict(torch.load(path))
  net.eval()
  net = net.to(device)
  return net

class VGG16(nn.Module):
  def __init__(self,model):
      super(VGG16,self).__init__()
      self.feature1 = model.features[:15]
      self.feature2 = model.features[15:]
      self.avgpool = model.avgpool
      self.classifier = model.classifier
      for module in self.feature1:
        if isinstance(module,nn.ReLU):
          module.inplace = False
      for module in self.feature2:
        if isinstance(module,nn.ReLU):
          module.inplace = False
        
  def forward(self,x):
    x = self.feature2(self.feature1(x))
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
if __name__=='__main__':
    # get_PACS_classifier(build=True)
    get_VLCS_classifier(build=True)

