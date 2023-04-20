import torch
import torch.nn as nn
from torchvision import models
from easydl import *

# BaseFeatureExtractor��̳���nn.Module�࣬��ʵ�������·�����
# forward��������Ϊģ���ǰ�򴫵ݷ�����������������������������������ǵ������
# __init__��������ʼ��ģ�顣
# output_num����������������ȡ���������������
# train����������ģ���ѵ��ģʽ��������������һ����ľ�ֵ�ͱ�׼��

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

# MyGradientReverseLayer ��һ���̳��� torch.autograd.Function �ľ�̬�����࣬
# ������������̬������forward �� backward���ֱ�ʵ����ǰ�򴫲��ͷ��򴫲���
# ��ǰ�򴫲��У�������� input ���浽 context �У������� input��
# �ڷ��򴫲��У���ǰ�򴫲������ grad_outputs ����һ��������-coeff����������None�ͽ����
# ����� coeff ��ʾ�ݶȷ�ת��ϵ����

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

# �� MyGradientReverseLayer ��װ��һ�� PyTorch ģ�飬ʹ��������� nn.Sequential �С�
# ��ģ��ĳ�ʼ���У���Ҫ����һ�� scheduler ���������ڼ��� coeff ��ֵ��
# ģ�����ѵ���������Զ�����ȫ�ֲ��� global_step �� coeff ��ֵ��
# ��ģ��� forward �����У�����ȫ�ֲ����� scheduler ����� coeff ��ֵ��
# Ȼ����� MyGradientReverseLayer ʵ���ݶȷ�ת�������ط�ת��Ľ����

# Usage�У�MyGradientReverseModule ���ڽ� x ��ת�ݶȣ��Ӷ�ʹ���ھ�����ת��������У�
# �ݶ����򴫲���֮ǰ�����磬����������Ӧѵ�������ʹ�� matplotlib �������ݶȷ�ת�����ߡ�

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

# ResNet50Fc��̳���BaseFeatureExtractor�࣬��ʵ�������·�����
# __init__��������ʼ��ResNet50ģ�ͣ����Լ���Ԥѵ��ģ�Ͳ����й�һ����
# forward������ʹ��ResNet50ģ����ȡͼ���������Ƚ��п�ѡ��������ǿ���������ѵ��ģʽ����
# Ȼ����й�һ�����������뵽ResNet50ģ���У���������������ƽ�����ء�
# output_num����������������ȡ���������������

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

# ����ResNext101�����������ȡ���ࣨBaseFeatureExtractor�����ࣩ������������ڽ������ͼ����ȡ������������
# �ڳ�ʼ�������У�����ṩ��Ԥѵ��ģ��·������ô�ͼ��ظ�·���µ�Ȩ���ļ�������Ĭ��ʹ��Ԥѵ����ResNext101ģ�͡�
# ����ṩ��Ԥѵ��ģ�ͻ�����Ҫ���������ݽ��й�һ������ô��ʹ��ImageNet�Ĺ�һ����ʽ��
# ��������ȡ�����forward��������һ��������������ͨ��������ػ��Ȳ�����ȡ��������ͼ������������������������չƽ��һά������
# output_num�������ظ�������ȡ�����������������ά�ȴ�С��

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

# ������һ����Ϊ "CLS" ��������ģ�ͣ���ģ�;��������㣺һ��ƿ���㣨bottleneck����һ��ȫ���Ӳ㣨fc������һ�� softmax �㡣
# �����ά��Ϊ in_dim�������ά��Ϊ out_dim�����У�ƿ�����ά���� bottle_neck_dim��
# �� forward �����У����Ƚ����� x �洢��һ���б� out �У�Ȼ�����α��� self.main �еĲ㣬��ÿһ��������ӵ��б� out �С�
# ��󣬽��б� out ���ء���������Ϊ���ڽ���ѵ����Ԥ��ʱ���ܹ�����ػ�ȡ��ÿһ���������Ա���к����Ĵ���������

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

# ������һ����Ϊadversarialnet��������ģ�ͣ����ڶԿ�ѵ���е��б����������̳���nn.Module�࣬
# ���ʼ������__init__������һ��������ȫ���Ӳ������ReLU������Լ�һ��Sigmoid�������ɵ����硣
# ���������һ����СΪin_feature�����������������һ��ֵ��[0,1]֮��ı�����������������Ϊ��ʵ���ݵĸ��ʡ�
# ������ṹ�л�ʹ����easydl���е�GradientReverseModule����ģ����Խ����򴫲������е��ݶȷ�ת������ʵ����������Ӧѵ���ķ�����
# forward�����������������ݸ�GradientReverseModule��main��ģ�����ǰ����㣬�������ģ�͵�Ԥ������

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