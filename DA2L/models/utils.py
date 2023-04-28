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

    return inputs_hat.detach(), max_value.detach()

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def get_source_share_weight(domain_out, hat, max_value, domain_temperature=1.0, class_temperature=1.0):
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

def get_target_share_weight(domain_out, hat, max_value, domain_temperature=1.0, class_temperature=1.0):
    return - get_source_share_weight(domain_out, hat, max_value, domain_temperature, class_temperature)

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()

def common_private_spilt(share_weight, feature):
    indices_private = torch.nonzero(torch.lt(share_weight, args.test.w_0))
    feature_private = torch.index_select(feature, 0, indices_private[:, 0])

    indices_common = torch.nonzero(torch.ge(share_weight, args.test.w_0))
    feature_common = torch.index_select(feature, 0, indices_common[:, 0])

    return feature_private, feature_common

def get_target_reuse_weight(reuse_out, fc, reuse_temperature=1.0):
    reuse_logit = reverse_sigmoid(reuse_out)
    reuse_logit = reuse_logit / reuse_temperature
    reuse_out = nn.Sigmoid()(reuse_logit)

    args.train.alpha
    return 

def pseudo_label_calibration(pslab, weight):
    weight = weight.transpose(1, 0).expand(pslab.shape[0], -1)
    weight = normalize_weight(weight)
    pslab = torch.exp(pslab)
    pslab = pslab * weight
    pslab = pslab / torch.sum(pslab, 1, keepdim=True)
    return pslab, weight

def get_source_reuse_weight(reuse_out, fc, w_avg, reuse_temperature=1.0, common_temperature = 1.0):
    reuse_logit = reverse_sigmoid(reuse_out)
    reuse_logit = reuse_logit / reuse_temperature
    reuse_out = nn.Sigmoid()(reuse_logit)

    label_ind = torch.nonzero(torch.ge(w_avg, args.test.w_0))
    fc = torch.index_select(fc, 1, label_ind[:, 0])
    fc = F.normalize(fc, p=1, dim=1)
    fc = TempScale(fc, 1000)
    fc_softmax = fc.softmax(1)
    max , _= fc_softmax.topk(2, dim=1, largest=True)
    class_tend = max[:,0]-max[:,1]
    class_tend = class_tend / torch.mean(class_tend)
    class_tend = reverse_sigmoid(class_tend)
    class_tend = class_tend / common_temperature
    class_tend = nn.Sigmoid()(class_tend)

    w_r = torch.var(reuse_out) / (torch.var(reuse_out)+torch.var(class_tend)) * reuse_out + \
    torch.var(class_tend) / (torch.var(reuse_out)+torch.var(class_tend)) * class_tend

    return w_r