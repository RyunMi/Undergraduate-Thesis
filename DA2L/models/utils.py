from easydl import *
import torch.nn.functional as F
from config.config import *

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def TempScale(p, t):
    return p / t

def perturb(inputs, before_softmax):
    softmax_output = before_softmax.softmax(1)
    softmax_output = TempScale(softmax_output, 0.5)
    max_value, max_target = torch.max(softmax_output, dim=1)
    xent = F.cross_entropy(softmax_output, max_target.long())
    d = torch.autograd.grad(xent, inputs)[0]
    d = torch.ge(d, 0)
    d = (d.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    d[0][0] = (d[0][0] )/(0.229)
    d[0][1] = (d[0][1] )/(0.224)
    d[0][2] = (d[0][2] )/(0.225)
    inputs_hat = torch.add(inputs.data, -args.train.eps, d)

    return inputs_hat, max_value

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def get_source_share_weight(domain_out, hat, max_value, domain_temperature=1.0, class_temperature=10.0):
    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)
    
    softmax_output_hat = hat.softmax(1)
    softmax_output_hat = TempScale(softmax_output_hat, 0.5)
    max_value_hat = torch.max(softmax_output_hat, dim=1).values
    pred_shift = torch.abs(max_value - max_value_hat).unsqueeze(1)
    min_val = pred_shift.min()
    max_val = pred_shift.max()
    pred_shift = (pred_shift - min_val) / (max_val - min_val)
    pred_shift = reverse_sigmoid(pred_shift)
    pred_shift = pred_shift / class_temperature
    pred_shift = nn.Sigmoid()(pred_shift)

    weight = domain_out - pred_shift
    weight = weight.detach()

    return weight


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)


def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()