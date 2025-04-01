import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import torchvision
from dataset_utils import get_preprocess
from attack_utils import Inverse_Norm
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import torch.nn.functional as F

def generate_grad_cam(images, model, target_layer):
    model.eval()
    batch_size = images.size(0)

    # 储存目标层的输出和梯度
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 注册钩子
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # 正向传播
    output = model(images)

    # 清零梯度
    model.zero_grad()

    # 反向传播预测最大的类
    pred_classes = output.argmax(dim=1)
    one_hot = torch.zeros_like(output)
    one_hot.scatter_(1, pred_classes.view(-1, 1), 1)
    one_hot.requires_grad_(True)
    output.backward(gradient=one_hot)

    # 移除钩子
    handle_forward.remove()
    handle_backward.remove()

    # 计算 Grad-CAM
    activations = activations[0].detach()
    gradients = gradients[0].detach()
    weights = gradients.mean(dim=[2, 3], keepdim=True)
    grad_cam = F.relu((weights * activations).sum(dim=1))
    grad_cam = F.interpolate(grad_cam.unsqueeze(1), (224, 224), mode='bilinear', align_corners=False)

    return grad_cam

def show_image_with_gradcam(image, grad_cam):
    # 将 Tensor 转化为 PIL 图片
    image = torchvision.transforms.ToPILImage()(image.cpu())
    # 将 Grad-CAM 转化为 PIL 图片
    grad_cam = torchvision.transforms.ToPILImage()(grad_cam.cpu())
    # 将 Grad-CAM 透明度降低并叠加到原始图片上
    heatmap = Image.blend(image.convert("RGBA"), grad_cam.convert("RGBA"), alpha=0.5)
    plt.imshow(heatmap)
    plt.show()
def top_30_percent_gradcam(gradcam):
    # 计算 70% 的阈值
    threshold = torch.quantile(gradcam.view(-1), 0.5)
    # 将小于阈值的元素置零
    gradcam = torch.where(gradcam > threshold, torch.ones_like(gradcam), torch.zeros_like(gradcam))
    return gradcam
if __name__ == "__main__":
    preprocess = get_preprocess()
    # 加载模型
    model = vgg16(pretrained=True)
    model.to(device)
    # 获取目标层（VGG16 的最后一个卷积层）
    target_layer = model.features[-1]
    dataset = torchvision.datasets.ImageFolder("../../../CDTA_datasets/birds_400/BIRDS-400/train",
                                      transform = preprocess)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    for x,_ in dataloader:
      x = x.to(device)
      gradcam = generate_grad_cam(x,model,target_layer)
      show_image_with_gradcam(x[0], gradcam[0])
      break
    







