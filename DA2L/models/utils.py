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

def perturb(inputs, feature_extractor, classifier):
    feature_extractor.eval()
    inputs.requires_grad = True
    features = feature_extractor.forward(inputs)
    _, _, score, _ = classifier.forward(features)
    softmax_score = TempScale(score, args.train.temp).softmax(1)
    max_value, max_target = torch.max(softmax_score, dim=1)
    xent = F.cross_entropy(softmax_score, max_target.detach().long())
    
    d = torch.autograd.grad(xent, inputs)[0]
    d = torch.ge(d, 0)
    d = (d.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    d[0][0] = (d[0][0] )/(0.229)
    d[0][1] = (d[0][1] )/(0.224)
    d[0][2] = (d[0][2] )/(0.225)
    inputs_hat = torch.add(inputs.data, -args.train.eps, d.detach())
    
    features_hat = feature_extractor.forward(inputs_hat)
    _, _, output_hat, _ = classifier.forward(features_hat)
    softmax_output_hat = TempScale(output_hat, args.train.temp).softmax(1)
    max_value_hat = torch.max(softmax_output_hat, dim=1).values
    pred_shift = torch.abs(max_value - max_value_hat).unsqueeze(1)
    feature_extractor.train()

    return pred_shift

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def get_source_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    # domain_logit = reverse_sigmoid(domain_out)
    # domain_logit = domain_logit / domain_temperature
    # domain_out = nn.Sigmoid()(domain_logit)
    
    min_val = pred_shift.min()
    max_val = pred_shift.max()
    pred_shift = (pred_shift - min_val) / (max_val - min_val)
    # pred_shift = reverse_sigmoid(pred_shift)
    # pred_shift = pred_shift / class_temperature
    # pred_shift = nn.Sigmoid()(pred_shift)

    weight = domain_out - pred_shift
    weight = weight.detach()

    return weight

def get_target_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    return - get_source_share_weight(domain_out, pred_shift, domain_temperature, class_temperature)

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

    return feature_private.detach(), feature_common.detach()

def get_target_reuse_weight(reuse_out, fc, reuse_temperature=1.0, common_temperature = 1.0):
    reuse_logit = reverse_sigmoid(reuse_out)
    reuse_logit = reuse_logit / reuse_temperature
    reuse_out = nn.Sigmoid()(reuse_logit)
    
    max , _= fc.topk(2, dim=1, largest=True)
    class_tend = max[:,0]-max[:,1]
    class_tend = class_tend / torch.mean(class_tend)
    # class_tend = reverse_sigmoid(class_tend)
    # class_tend = class_tend / common_temperature
    # class_tend = nn.Sigmoid()(class_tend)
    
    w_r1 = torch.var(reuse_out)
    w_r2 = torch.var(class_tend)
    w_r1 = torch.where(torch.isnan(w_r1), torch.full_like(w_r1, 0), w_r1)
    w_r2 = torch.where(torch.isnan(w_r2), torch.full_like(w_r2, 0), w_r2)

    if w_r1 or w_r2:
        w_r = (w_r1 / (w_r1 + w_r2) * reuse_out).view(-1) + \
    (w_r2 / (w_r1 + w_r2) * class_tend).view(-1)
    else: 
        w_r = reuse_out.view(-1) + class_tend.view(-1)
    w_r = w_r.detach()

    return w_r

def compute_avg_weight(weight, label, class_weight):
    for i in range(len(class_weight)):
        mask = (label == i)
        class_weight[i] = weight[mask].mean()
    class_weight = torch.where(torch.isnan(class_weight), torch.full_like(class_weight, 0), class_weight)
    return class_weight


def pseudo_label_calibration(pslab, weight):
    #weight = weight.transpose(1, 0).expand(pslab.shape[0], -1)
    weight = normalize_weight(weight)
    pslab = torch.exp(pslab)
    pslab = pslab * weight
    pslab = pslab / torch.sum(pslab, 1, keepdim=True)
    return pslab, weight.detach()

def get_source_reuse_weight(reuse_out, fc, w_avg, reuse_temperature=1.0, common_temperature = 1.0):
    reuse_logit = reverse_sigmoid(reuse_out)
    reuse_logit = reuse_logit / reuse_temperature
    reuse_out = nn.Sigmoid()(reuse_logit)

    label_ind = torch.nonzero(torch.ge(w_avg, args.test.w_0))
    fc = torch.index_select(fc, 1, label_ind[:, 0])
    fc = F.normalize(fc, p=1, dim=1)
    fc = TempScale(fc, args.train.temp)
    fc_softmax = fc.softmax(1)

    if min(fc_softmax.shape) == 0:
        class_tend = torch.zeros((fc_softmax.shape[0]),1)
    
    if fc_softmax.shape[1] == 1:
        class_tend = fc_softmax
    else:
        max , _= fc_softmax.topk(2, dim=1, largest=True)
        class_tend = max[:,0]-max[:,1]
        class_tend = class_tend / torch.mean(class_tend)
    
    # class_tend = reverse_sigmoid(class_tend)
    # class_tend = class_tend / common_temperature
    # class_tend = nn.Sigmoid()(class_tend)

    w_r1 = torch.var(reuse_out)
    w_r2 = torch.var(class_tend)
    w_r1 = torch.where(torch.isnan(w_r1), torch.full_like(w_r1, 0), w_r1)
    w_r2 = torch.where(torch.isnan(w_r2), torch.full_like(w_r2, 0), w_r2)

    if w_r1 or w_r2:
        w_r = (w_r1 / (w_r1 + w_r2) * reuse_out).view(-1) + \
    (w_r2 / (w_r1 + w_r2) * class_tend).view(-1)
    else: 
        w_r = reuse_out.view(-1) + class_tend.view(-1)
    w_r = w_r.detach()

    return w_r