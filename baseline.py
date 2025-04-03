from BIA.eval import netG
import torch
import torchvision
from eval_utils import get_dataset,get_dataset_path,get_args,get_model
from attack_utils import Norm,input_diversity,gkern,TI_gradient,project,FIA_map,Inverse_Norm
from CDA.generators import GeneratorResnet
from CDTA.utils.cdta import CDTAttack
from CDTA.utils.encoder import AttackEncoder
from model_utils import prepare_mae_model,prepare_classifier
from dataset_utils import get_dataloader
import os
from pixel_models.models import ImDecoder,ImEncoder
from vq_vae_2_pytorch.vqvae import VQVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Pixel_VAE():
  #(64,64) 32 0'./cache',64
    encoder = ImEncoder(in_size=(64, 64), zsize=32, depth=0, colors='./cache')
    decoder = ImDecoder(in_size=(64, 64), zsize=32, depth=0, out_channels=64)
    pixel_vae = torch.nn.Sequential([encoder,decoder])
    return pixel_vae

def VQ_VAE_2():
    return VQVAE()

def FDA_loss(x_adv, x_clean, model, layer_name):
  mid_output, mid_grad = None, None
  model.eval()

  def get_mid_output(m, i, o):
      nonlocal mid_output
      mid_output = o.clone()

  def get_mid_grad(m, i, o):
      nonlocal mid_grad
      mid_grad = o[0].clone() 
  
  model(x_clean)
  mid_output_clean = mid_output.clone()
  model(x_adv)
  mid_output_adv = mid_output.clone()
  term_1 = torch.norm(mid_output_adv * (mid_output_clean < mid_output_clean.mean(dim=1)))
  term_2 = torch.norm(mid_output_adv * (mid_output_clean > mid_output_clean.mean(dim=1)))
  loss = torch.log(term_1) - torch.log(term_2)
  return loss

def TAP_loss(logits, label, x, x_adv, original_mids, new_mids):
    """
    Overriden for TAP
    """
    yita = 0.01
    lam = 0.005
    alpha_tap = 0.5
    s = 3
    l1 = torch.nn.CrossEntropyLoss()(logits, label)
    l2 = 0.
    for i, new_mid in enumerate(new_mids):
        a = torch.sign(original_mids[i] ) * torch.pow(torch.abs(original_mids[i]),alpha_tap)
        b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid),alpha_tap)
        l2 += lam * (a-b).norm() **2
    l3 = yita * torch.abs(torch.nn.AvgPool2d(s)(x-x_adv)).sum()
    return l1 + l2 + l3

def NRDM_loss(x, x_adv, model, layer_name):
    mid_output, mid_grad = None, None
    model.eval()

    def get_mid_output(m, i, o):
        nonlocal mid_output
        mid_output = o.clone()

    def get_mid_grad(m, i, o):
        nonlocal mid_grad
        mid_grad = o[0].clone()

    model(x)
    mid_output_clean = mid_output.clone()
    model(x_adv)
    mid_output_adv = mid_output.clone()
    loss = torch.norm(mid_output_clean - mid_output_adv)
    return loss

 

def BIA(RN=False,DA=False):
  G = netG(RN=RN,DA=DA)
  G.to(device)
  for param in G.parameters():
    param.requires_grad = False

  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      model = prepare_classifier(name,model_type)
      model.to(device)
      for param in model.parameters():
        param.requires_grad = False
      step = 0
      fool = 0; total = 0;
      for x,label in dataloader:
        x = x.to(device); label = label.to(device)
        x1 = Inverse_Norm(x.clone())
        pred_clean = model(x)
        adv = G(x1)
        adv = torch.min(torch.max(adv, x1 - 16./255), x1 + 16./255)
        adv = torch.clamp(adv, 0.0, 1.0)
        adv_x = Norm(adv)
        pred_adv = model(adv_x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step >= 20:
          print(f"{name} {model_type} fool_rate:{fool/total}")
          break

def MAE(filename):
  G = prepare_mae_model(model_type="gan")
  if os.path.exists(filename):  
    G.load_state_dict(torch.load(filename))
    print(f"loading from {filename}")
  G.to(device)
  G.eval()
  for param in G.parameters():
    param.requires_grad = False
  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      model = prepare_classifier(name,model_type)
      model.eval(); model.to(device)
      for param in model.parameters():
        param.requires_grad = False
      step = 0
      fool = 0; total = 0;
      for x,label in dataloader:
        x = x.to(device); label = label.to(device)
        pred_clean = model(x)
        _,decoded_tokens,_ = G(x,0)
        adv_x = G.unpatchify(decoded_tokens)
        adv_x = project(x,adv_x)
        pred_adv = model(adv_x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step >= 10:
          print(f"{name} {model_type} fool_rate:{fool/total}")
          break
  G.cpu()

def MI_FGSM():
  # 加载 VGG16 模型
  source_model = torchvision.models.inception_v3(pretrained=True)
  if torch.cuda.is_available():
    source_model = source_model.cuda()

  source_model.eval()
  
  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      target_model = prepare_classifier(name,model_type)
      if torch.cuda.is_available():
        target_model = target_model.cuda()

      target_model.eval()

      step = 0
      fool = 0
      total = 0
      for x, label in dataloader:
        if torch.cuda.is_available():
          x = x.cuda()
          label = label.cuda()
        
        # 初始化动量
        momentum = torch.zeros_like(x)
        adv_x = x.clone()
        adv_x.requires_grad = True
        # 迭代生成对抗样本
        for _ in range(10):
          output = source_model(adv_x)
          y = output.argmax(dim=1)
          loss = torch.nn.functional.cross_entropy(output, y)
          source_model.zero_grad()
          loss.backward()

          grad = adv_x.grad.data
          grad_norm = torch.nn.functional.normalize(grad)

          # 更新动量
          momentum = momentum * 0.9 + grad_norm
          
          # 更新对抗样本
          adv_x = adv_x + 0.03425 * torch.sign(momentum)
          adv_x = project(x, adv_x)
          
          adv_x = adv_x.detach()
          adv_x.requires_grad = True

        output_adv = target_model(adv_x)
        pred_adv = output_adv.argmax(dim=1)

        output_clean = target_model(x)
        pred_clean = output_clean.argmax(dim=1)

        fool += ((pred_adv != label) & (pred_clean == label)).sum().item()
        total += (pred_clean == label).sum().item()

        step += 1
        if step >= 20:
          print(f"{name} {model_type} fool_rate: {fool / total} total:{total}")
          break


def TI_DIM():
    # 加载 VGG16 模型
    source_model = torchvision.models.resnet152(pretrained=True)
    for param in source_model.parameters():
      param.requires_grad = False
    if torch.cuda.is_available():
        source_model = source_model.cuda()

    source_model.eval()

    names = ["comic_books","oxford_102_flower","birds_400","food_101"]
    model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]

    for name in names:
        for model_type in model_types:
            dataloader = get_dataloader(name=name)
            target_model = prepare_classifier(name,model_type)
            if torch.cuda.is_available():
                target_model = target_model.cuda()

            target_model.eval()

            step = 0
            fool = 0
            total = 0
            for x, label in dataloader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    label = label.cuda()
                adv_x = x.clone()
                adv_x.requires_grad = True

                # 初始化动量
                momentum = torch.zeros_like(x)

                # 迭代生成对抗样本
                for _ in range(10):
                    # 应用输入多样性
                    x_diversity = input_diversity(adv_x)

                    output = source_model(x_diversity)
                    y = output.argmax(dim=1)
                    loss = torch.nn.functional.cross_entropy(output, y)

                    source_model.zero_grad()
                    loss.backward()

                    grad = adv_x.grad.data
                    grad_norm = torch.nn.functional.normalize(grad)

                    # 应用平移不变性
                    ti_grad = TI_gradient(grad_norm)
                    # 更新动量
                    momentum = momentum * 0.9 + ti_grad

                    # 更新对抗样本
                    adv_x = adv_x + 0.03425 * torch.sign(momentum)
                    adv_x = project(x, adv_x)

                    adv_x = adv_x.detach()
                    adv_x.requires_grad = True

                output_adv = target_model(adv_x)
                pred_adv = output_adv.argmax(dim=1)

                output_clean = target_model(x)
                pred_clean = output_clean.argmax(dim=1)

                fool += ((pred_adv != label) & (pred_clean == label)).sum().item()
                total += (pred_clean == label).sum().item()

                step += 1
                if step >= 20:
                    fool_rate = fool/total
                    print(f"{name} {model_type} fool_rate: {fool_rate:.4f}")
                    break

def FIA():
  def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
  # 加载 VGG16 模型
  source_model = torchvision.models.resnet152(pretrained=True)
  if torch.cuda.is_available():
    source_model = source_model.cuda()

  source_model.eval()
  
  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  layer_name = "layer2"
  
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      target_model = prepare_classifier(name,model_type)
      if torch.cuda.is_available():
        target_model = target_model.cuda()

      target_model.eval()

      step = 0
      fool = 0
      total = 0
      for x, label in dataloader:
        if torch.cuda.is_available():
          x = x.cuda()
          label = label.cuda()
        adv_x = x.clone()
        
        # 初始化动量
        momentum = torch.zeros_like(x)
        y = 0
        with torch.no_grad():
          output = source_model(x)
          y = output.argmax(dim=1)
        Map = FIA_map(x,y,source_model,layer_name)
        # 迭代生成对抗样本
        for _ in range(10):
          adv_x.requires_grad = True
          source_model._modules.get(layer_name).register_forward_hook(get_mid_output)
          source_model(adv_x)
          loss = -(Map*mid_output).sum()
          loss.backward()

          grad = adv_x.grad.data
          grad_norm = torch.nn.functional.normalize(grad)

          # 更新动量
          momentum = momentum * 0.9 + grad_norm
          
          # 更新对抗样本
          adv_x = adv_x + 0.03425 * torch.sign(momentum)
          adv_x = project(x, adv_x)
          
          adv_x = adv_x.detach()
          adv_x.requires_grad = True

        output_adv = target_model(adv_x)
        pred_adv = output_adv.argmax(dim=1)

        output_clean = target_model(x)
        pred_clean = output_clean.argmax(dim=1)

        fool += ((pred_adv != label) & (pred_clean == label)).sum().item()
        total += (pred_clean == label).sum().item()

        step += 1
        if step >= 20:
          print(f"{name} {model_type} fool_rate: {fool / total}")
          break

def CDA():
  G = GeneratorResnet()
  # G = Pixel_VAE()
  # G = VQ_VAE_2()
  G.load_state_dict(torch
        .load(f"CDA/saved_models/netG_-1_img_res152_imagenet_0_rl.pth"))
  G.to(device)
  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      model = prepare_classifier(name,model_type)
      step = 0
      fool = 0; total = 0;
      
      for x,label in dataloader:
        x = x.to(device); label = label.to(device)
        with torch.no_grad():
          pred_clean = model(x)
          adv_x = G(Inverse_Norm(x))
          adv_x = project(x,Norm(adv_x))
          pred_adv = model(adv_x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step >= 20:
          print(f"{name} {model_type} fool_rate:{fool/total}")
          break

def CDTA():
  names = ["comic_books","oxford_102_flower","birds_400","food_101"]
  model_types = ["resnet34","densenet161","inception_v3","vgg16_bn"]
  for name in names:
    for model_type in model_types:
      dataloader = get_dataloader(name=name)
      model = prepare_classifier(name,model_type)
      step = 0
      fool = 0; total = 0;
      
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
        pred_clean = model(x)
        pred_adv = model(adv_x)
        fool += ((pred_adv.argmax(dim=1)!=label)*(pred_clean.argmax(dim=1)==label)).sum()
        total += (pred_clean.argmax(dim=1)==label).sum()
        step += 1
        if step >=20:
          print(f"{name} {model_type} fool_rate:{fool/total}")
          break


def train_baseline(dataloader, epochs, steps_per_epoch, classifier, MAE_model=None, filename="MAE.pth",
                    ):
    classifier.eval()
    classifier.to(device)

    if MAE_model is None:
        MAE_model = prepare_mae_model(model_type="gan")
    MAE_model.shuffle = False
    MAE_model.to(device)
    optimizer = torch.optim.Adam(MAE_model.parameters(),
                                 lr=1e-5)  # define an optimizer for the "sharpness-aware" update

    for blk in MAE_model.blocks:
        for param in blk.parameters():
                param.requries_grad = False

    MAE_model.shuffle = False

    if os.path.exists(filename):
        print(f"loading from {filename}")
        MAE_model.load_state_dict(torch.load(filename))

    for param in classifier.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        step = 0
        for x, label in dataloader:
            optimizer.zero_grad()
            x = x.to(device);
            label = label.to(device).squeeze()

            _, decoded_token, _ = MAE_model(x, 0)

            adv_x = MAE_model.unpatchify(decoded_token)
            adv_x = project(x, adv_x)

            loss = FDA_loss(adv_x, x, classifier, "features");
            # TAP
            # logits = classifier.forward(x)
            # original_mids = FIA_map(x, label, classifier, "features", 1, 0)
            # new_mids = FIA_map(adv_x, label, classifier, "features", 1, 0)
            # loss = TAP_loss(logits, labels, adv_x, x, original_mids, new_mids)

            # NRDM
            # loss = NRDM_loss(x, adv_x, classifier, "features")
            pred_adv = classifier(adv_x)
            pred_clean = classifier(x)
            fool = 0;
            total = 0
            fool += ((pred_adv.argmax(dim=1) != label) * (pred_clean.argmax(dim=1) == label)).sum()
            total += (pred_clean.argmax(dim=1) == label).sum()
            fool_rate = fool / total
            loss.backward()
            optimizer.step()
            suc_rate = (pred_adv.argmax(dim=1) != label).sum() / x.shape[0]

            print(f"epoch {epoch} step {step}:\
loss:{loss:.2f} fool_rate:{fool_rate:.3f} total:{total}")
            step += 1
            if (step >= steps_per_epoch):
                break
        torch.save(MAE_model.state_dict(), filename)
        print(f"Model saved to {filename}")

if __name__ == "__main__":
  MAE("mae_models/CMAE_vgg16_timm.pth")
