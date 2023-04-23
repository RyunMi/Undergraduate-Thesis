import torch
import torch.nn as nn
from torchvision import models
from easydl import *
from torch.autograd import Function
from typing import  List, Dict, Optional, Any, Tuple

# BaseFeatureExtractor类继承了nn.Module类，并实现了以下方法：
# forward方法：作为模块的前向传递方法，接收任意数量的输入参数并返回它们的输出。
# __init__方法：初始化模块。
# output_num方法：返回特征提取器的输出特征数。
# train方法：设置模块的训练模式，并冻结批量归一化层的均值和标准差

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

# ResNet50Fc类继承了BaseFeatureExtractor类，并实现了以下方法：
# __init__方法：初始化ResNet50模型，可以加载预训练模型并进行归一化。
# forward方法：使用ResNet50模型提取图像特征，先进行可选的数据增强（如果处于训练模式），
# 然后进行归一化并传递输入到ResNet50模型中，最后将输出的特征拉平并返回。
# output_num方法：返回特征提取器的输出特征数。

class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

# 基于ResNext101网络的特征提取器类（BaseFeatureExtractor的子类），该类可以用于将输入的图像提取出特征向量。
# 在初始化函数中，如果提供了预训练模型路径，那么就加载该路径下的权重文件，否则默认使用预训练的ResNext101模型。
# 如果提供了预训练模型或者需要对输入数据进行归一化，那么就使用ImageNet的归一化方式。
# 该特征提取器类的forward函数接收一个输入张量，并通过卷积、池化等操作提取出该输入图像的特征向量，最后将特征向量展平成一维张量。
# output_num函数返回该特征提取器的输出特征向量的维度大小。

class Resnext101Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, model_path=None, normalize=True):
        super(Resnext101Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnext = models.resnext101_32x8d(pretrained=False)
                self.model_resnext.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnext = models.resnext101_32x8d(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnext = self.model_resnext
        self.conv1 = model_resnext.conv1
        self.bn1 = model_resnext.bn1
        self.relu = model_resnext.relu
        self.maxpool = model_resnext.maxpool
        self.layer1 = model_resnext.layer1
        self.layer2 = model_resnext.layer2
        self.layer3 = model_resnext.layer3
        self.layer4 = model_resnext.layer4
        self.avgpool = model_resnext.avgpool
        self.__in_features = model_resnext.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

# 定义了一个名为 "CLS" 的神经网络模型，该模型具有三个层：一个瓶颈层（bottleneck），一个全连接层（fc），和一个 softmax 层。
# 输入的维度为 in_dim，输出的维度为 out_dim。其中，瓶颈层的维度是 bottle_neck_dim。
# 在 forward 函数中，首先将输入 x 存储在一个列表 out 中，然后依次遍历 self.main 中的层，将每一层的输出添加到列表 out 中。
# 最后，将列表 out 返回。这样做是为了在进行训练或预测时，能够方便地获取到每一层的输出，以便进行后续的处理或分析。

class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

# 定义了一个名为AdversarialNetwork的神经网络模型，用于对抗训练中的判别器任务。它继承了nn.Module类，
# 其初始化函数__init__定义了一个由三个全连接层和两个ReLU激活函数以及一个Sigmoid激活函数组成的网络。
# 该网络接受一个大小为in_feature的输入向量，并输出一个值在[0,1]之间的标量，代表输入样本为真实数据的概率。
# 在网络结构中还使用了easydl库中的GradientReverseModule，该模块可以将反向传播过程中的梯度反转，用于实现领域自适应训练的方法。
# 与TOOR flip_coeff相同（除了没有预训练5000步且最大迭代次数不为400000）：
# coeff=0.0 + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (1.0 - 0.0)
# forward函数将输入向量传递给GradientReverseModule和main子模块进行前向计算，最终输出模型的预测结果。

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y
    
# 自定义梯度反转层、域判别器以及域对抗损失（未修改）
# class GradientReverseFunction(Function):

#     @staticmethod
#     def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
#         ctx.coeff = coeff
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
#         return grad_output.neg() * ctx.coeff, None

# class WarmStartGradientReverseLayer(nn.Module):

#     def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
#                  max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
#         super(WarmStartGradientReverseLayer, self).__init__()
#         self.alpha = alpha
#         self.lo = lo
#         self.hi = hi
#         self.iter_num = 0
#         self.max_iters = max_iters
#         self.auto_step = auto_step

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """"""
#         coeff = np.float(
#             2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
#             - (self.hi - self.lo) + self.lo
#         )
#         if self.auto_step:
#             self.step()
#         return GradientReverseFunction.apply(input, coeff)

#     def step(self):
#         """Increase iteration number :math:`i` by 1"""
#         self.iter_num += 1

# def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
#     """Computes the accuracy for binary classification"""
#     with torch.no_grad():
#         batch_size = target.size(0)
#         pred = (output >= 0.5).float().t().view(-1)
#         correct = pred.eq(target.view(-1)).float().sum()
#         correct.mul_(100. / batch_size)
#         return correct

# class DomainDiscriminator(nn.Module):

#     def __init__(self, in_feature: int, hidden_size: int):
#         super(DomainDiscriminator, self).__init__()
#         self.layer1 = nn.Linear(in_feature, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.relu1 = nn.ReLU()
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.BatchNorm1d(hidden_size)
#         self.relu2 = nn.ReLU()
#         self.layer3 = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """"""
#         x = self.relu1(self.bn1(self.layer1(x)))
#         x = self.relu2(self.bn2(self.layer2(x)))
#         y = self.sigmoid(self.layer3(x))
#         return y

#     def get_parameters(self) -> List[Dict]:
#         return [{"params": self.parameters(), "lr_mult": 1.}]

# class DomainAdversarialLoss(nn.Module):

#     def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean'):
#         super(DomainAdversarialLoss, self).__init__()
#         self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
#         self.domain_discriminator = domain_discriminator
#         self.bce = nn.BCELoss(reduction=reduction)
#         self.domain_discriminator_accuracy = None

#     def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, w_s, w_t) -> torch.Tensor:
#         f = self.grl(torch.cat((f_s, f_t), dim=0))
#         d = self.domain_discriminator(f)
#         d_s, d_t = d.chunk(2, dim=0)
#         d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
#         d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
#         self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
#         source_loss = torch.mean(w_s * self.bce(d_s, d_label_s).view(-1))
#         target_loss = torch.mean(w_t * self.bce(d_t, d_label_t).view(-1))
#         return 0.5 * (source_loss + target_loss)