import torch
from torch import nn
import torch.nn.functional as F
from mae_pytorch import models_vit
from mae_pytorch.util import misc
from dataset_utils import get_preprocess
import argparse
import torchvision
from attack_utils import TI_attack
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def distillation_loss(y_pred_student, y_pred_teacher, y_true, temperature=1, alpha=0.5):
    kl_loss = F.kl_div(F.log_softmax(y_pred_student / temperature, dim=-1), F.softmax(y_pred_teacher / temperature, dim=-1), reduction='batchmean')
    ce_loss = F.cross_entropy(y_pred_student, y_true)
    return alpha * kl_loss + (1 - alpha) * ce_loss

def run(S_model,filename,epochs=40,steps_per_epoch=1000,batch_size=50):
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
  preprocess = get_preprocess()
  if os.path.exists(filename):
    S_model.load_state_dict(torch.load(filename))
    print(f"loading from {filename}")

  T_model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
  T_model.eval(); T_model.to(device)
  S_model.train(); S_model.to(device)
  for param in T_model.parameters():
    param.requires_grad = False
  misc.load_model(args,T_model,None,None)
  
  
  dataset = torchvision.datasets.ImageFolder("ImageNet",transform=preprocess)
  dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,prefetch_factor=4,num_workers=2,shuffle=True)
  optimizer = torch.optim.Adam(S_model.parameters(),lr=1e-5)

  for epoch in range(epochs):
    step = 0
    S_model.train()
    for x,labels in dataloader:
        x = x.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        T_pred = T_model(x)
        S_pred = S_model(x)
        loss = distillation_loss(S_pred,T_pred,labels)
        loss.backward()
        optimizer.step()
        acc = (S_pred.argmax(dim=1) == labels).sum() / x.shape[0]
        print(f"epoch {epoch} step {step}, loss:{loss}, acc:{acc}")
        step += 1
        if step%100 == 0:
          torch.save(S_model.state_dict(),filename)
        
    print(f"S_model saved to {filename}")

if __name__ == '__main__':
  resnet50 = torchvision.models.resnet50(pretrained=True)
  filename = "CNN/resnet50.pth"
  run(resnet50,filename)
    





  