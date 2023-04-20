import torch
import torch.nn as nn
from torchvision import models
from easydl import *

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

# MyGradientReverseLayer 是一个继承了 torch.autograd.Function 的静态方法类，
# 定义了两个静态方法：forward 和 backward，分别实现了前向传播和反向传播。
# 在前向传播中，将传入的 input 保存到 context 中，并返回 input。
# 在反向传播中，将前向传播传入的 grad_outputs 乘以一个常数（-coeff），并返回None和结果。
# 这里的 coeff 表示梯度反转的系数。

class MyGradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer.apply
        y = grl(0.5, x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return ``input``, ``backward`` will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

# 将 MyGradientReverseLayer 封装成一个 PyTorch 模块，使其可以用在 nn.Sequential 中。
# 在模块的初始化中，需要传入一个 scheduler 函数，用于计算 coeff 的值。
# 模块会在训练过程中自动更新全局步数 global_step 和 coeff 的值。
# 在模块的 forward 方法中，根据全局步数和 scheduler 计算出 coeff 的值，
# 然后调用 MyGradientReverseLayer 实现梯度反转，并返回反转后的结果。

# Usage中：MyGradientReverseModule 用于将 x 反转梯度，从而使其在经过反转后的网络中，
# 梯度逆向传播到之前的网络，进行域自适应训练。最后使用 matplotlib 绘制了梯度反转的曲线。

class MyGradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(MyGradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.register_buffer('global_step', torch.zeros(1))
        self.coeff = 0.0
        self.grl = MyGradientReverseLayer.apply

    def forward(self, x):
        if self.global_step.item() < 25000:
            self.coeff = 0
        else:
            self.coeff = self.scheduler(self.global_step.item() - 25000)
        if self.training:
            self.global_step += 1.0
        return self.grl(self.coeff, x)

# ResNet50Fc类继承了BaseFeatureExtractor类，并实现了以下方法：
# __init__方法：初始化ResNet50模型，可以加载预训练模型并进行归一化。
# forward方法：使用ResNet50模型提取图像特征，先进行可选的数据增强（如果处于训练模式），
# 然后进行归一化并传递输入到ResNet50模型中，最后将输出的特征拉平并返回。
# output_num方法：返回特征提取器的输出特征数。

class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self, transform=None, model_path=None, normalize=True):
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
        self.transform=transform

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
        if self.training and self.transform is not None:
            x = self.transform(x)
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
    def __init__(self, transform=None, model_path=None, normalize=True):
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
        self.transform=transform

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
        if self.training and self.transform is not None:
            x = self.transform(x)
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

# 定义了一个名为adversarialnet的神经网络模型，用于对抗训练中的判别器任务。它继承了nn.Module类，
# 其初始化函数__init__定义了一个由三个全连接层和两个ReLU激活函数以及一个Sigmoid激活函数组成的网络。
# 该网络接受一个大小为in_feature的输入向量，并输出一个值在[0,1]之间的标量，代表输入样本为真实数据的概率。
# 在网络结构中还使用了easydl库中的GradientReverseModule，该模块可以将反向传播过程中的梯度反转，用于实现领域自适应训练的方法。
# forward函数将输入向量传递给GradientReverseModule和main子模块进行前向计算，最终输出模型的预测结果。

class adversarialnet(nn.Module):
    """
    Discriminator network.
    """
    def __init__(self, in_feature):
        super(adversarialnet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16,16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=100000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y