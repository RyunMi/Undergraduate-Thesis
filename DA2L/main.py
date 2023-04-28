from data.data_loader import *
from models.net import *
from models.utils import *
import datetime#���ڴ�������ʱ��
from tqdm import tqdm#�����������н�������ʾ������
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
    #�����жϴ����Ƿ��� Jupyter Notebook �����У���������Ӧ�Ľ�����ģʽ��
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

# cudnn.benchmark = True �� cudnn.deterministic = True �ֱ����úͽ����� cudnn ���Զ��������ƺ�����ԣ������ѵ���ٶȺͽ�����ȶ��ԡ�
# ���� args.misc.gpus ��ֵ���ж��Ƿ�ʹ�� GPU ���㡣��� args.misc.gpus С�� 1��˵����ʹ�� GPU ���㣬
# ��ʱ���������� CUDA_VISIBLE_DEVICES ����Ϊ���ַ�������������豸 output_device ��Ϊ CPU��
# ���򣬵��� select_GPUs(args.misc.gpus) ����ѡ�� args.misc.gpus �����õ� GPU��������һ�� GPU ��Ϊ����豸 output_device��
# ʹ�õ�ǰʱ������һ����־Ŀ¼ log_dir��
# ����һ�� SummaryWriter ʵ�� logger�����ڼ�¼ѵ�����̺ͽ���������䱣���� log_dir Ŀ¼�¡�
# ����ǰ���� save_config д��һ�� YAML ��ʽ�������ļ��������� log_dir Ŀ¼�µ� config.yaml �ļ��С�

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'resnext101':Resnext101Fc
}

# ������һ����Ϊ TotalNet ��������ģ�ͣ����������ĸ����л�����ģ�ͣ�feature_extractor��classifier��discriminator 
# �� discriminator_separate����
# model_dict ��һ���ֵ䣬����������ֵ�ԣ���Ӧ��ͬ��Ԥѵ��ģ�ͣ�'resnet50' ��Ӧ���� ResNet50Fc ģ�ͣ�
# 'resnext101' ��Ӧ���� Resnext101Fc ģ�͡�
# TotalNet �̳��� nn.Module �࣬�����乹�캯���ж������ĸ���ģ�ͣ�feature_extractor��classifier��discriminator 
# �� discriminator_separate���ֱ����������ȡ�������������б����ͷ����б�����
# ���У�feature_extractor ʹ�� args.model.base_model ��ָ����Ԥѵ��ģ�ͣ�classifier ʹ���Զ���� CLS �࣬
# discriminator �� discriminator_separate ʹ���Զ���� AdversarialNetwork �ࡣ
# forward ����������ǰ�������̣�����ͨ��������ȡ���õ�������ʾ f��Ȼ��ͨ���������õ������� y �͸������� _ �� __��
# ���ͨ���б����ͷ����б����ֱ�õ��Կ���ʧ d �� d_0������ y��d �� d_0 ��Ϊ������ء�
# totalNet �� TotalNet ��һ��ʵ����
# feature_extractor��classifier��discriminator �� discriminator_separate ����ʹ�� nn.DataParallel ���в��л��������ģ�͡�
# ���У�device_ids ����ָ����ʹ�õ� GPU �豸 ID��output_device ����ָ�������������ڵ��豸��������ģ�;�ʹ�� GPU ����ѵ����train(True)����

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.domain_discriminator = AdversarialNetwork(256)
        self.reuse_discriminator_s = AdversarialNetwork(256)
        self.reuse_discriminator_t = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.domain_discriminator(_)
        hat_d_s = self.reuse_discriminator_s(_)
        hat_d_t = self.reuse_discriminator_t(_)
        return y, d, hat_d_s, hat_d_t


totalNet = TotalNet()

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
domain_discriminator = nn.DataParallel(totalNet.domain_discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
reuse_discriminator_s = nn.DataParallel(totalNet.reuse_discriminator_s, device_ids=gpu_ids, output_device=output_device).train(True)
reuse_discriminator_t = nn.DataParallel(totalNet.reuse_discriminator_t, device_ids=gpu_ids, output_device=output_device).train(True)

# ��������в���args.test.test_onlyΪTrue����ô��������Ԥѵ��ģ�͵�Ȩ���ļ���Ȼ��ʹ�ø�ģ�ͶԲ��Լ�����Ԥ�⣬������ģ�͵Ĳ���׼ȷ�ʡ�
# ���ȣ������ͨ��torch.load��������Ԥѵ��ģ�͵�Ȩ���ļ�����ʹ��load_state_dict������Ȩ���ļ��еĲ���ֵ���ص���Ӧ������ģ���У�
# ����feature_extractor��classifier��discriminator��discriminator_separate��
# Ȼ�󣬴���ᴴ��һ������Ϊlen(source_classes) + 1���б�counters������source_classes��Դ����������
# ������������ᴴ��һ��TrainingModeManager���󣬽���������ģ��(feature_extractor��classifier��discriminator_separate)
# ��train��������ΪFalse�������Ϳ����ڲ���������Ȩ�ص�����½��в��ԡ�Ȼ�󣬴���ᴴ��һ��Accumulator����target_accumulator��
# �����ۻ�ÿ��������feature��predict_prob��label��domain_prob��before_softmax��target_share_weight�����ݣ���Щ���ݻ�������ļ������õ���
# ���ţ������ͨ��һ��forѭ�����Բ��Լ��е�ÿ��ͼ�����Ԥ�⡣����ÿ��ͼ�񣬴���Ὣ��ŵ�GPU�ϣ�������feature_extractor
# ��classifier�����ǰ�򴫲����㣬�õ���������feature���������predict_prob����ʵ��ǩlabel���������domain_prob��δ��softmax�����
# before_softmax��Ȼ�󣬴��������domain_prob��before_softmax����ÿ��Ŀ��������target_share_weight����ʾ������Դ���Ŀ����Ĺ��׳̶ȡ�
# �ڵõ���Щ���ݺ󣬴����ͨ��globals()��������Щ����ת��Ϊȫ�ֱ�����������target_accumulator��
# ������������ᴴ��һ����Ϊoutlier�ĺ������ú�������һ��each_target_share_weight������
# ���each_target_share_weightС����ֵargs.test.w_0���򷵻�True�����򷵻�False��
# Ȼ�󣬴���ᴴ��һ������Ϊlen(source_classes) + 1��counters�б�����counters[i]��ʾԴ�������Ϊi����������ȷ��������������
# ���ţ������ͨ��һ��forѭ������ÿ��Ԥ���������each_predict_prob����ʵ��ǩeach_label��Ŀ��������target_share_weight���б�����
# ���each_label��Դ���У�������ȷ������������������Ӧ��counters�С����each_label����Դ���У�������ȷ����������������counters[-1]�С�
# ��󣬴�������ÿ��counters��׼ȷ��

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    domain_discriminator.load_state_dict(data['domain_discriminator'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier, domain_discriminator], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax',
                         'target_share_weight']) as target_accumulator, \
            torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature)
            
            domain_prob = domain_discriminator.forward(__)

            hat, max_value = perturb(im, before_softmax)

            hat_feature = feature_extractor.forward(hat)

            _, _, hat_before_softmax, _ = classifier.forward(hat_feature)

            target_share_weight = get_target_share_weight(domain_prob, hat_before_softmax,  max_value,
                                                                  domain_temperature=1.0, class_temperature=1.0)

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    def outlier(each_target_share_weight):
        return each_target_share_weight < args.test.w_0

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

    for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label, target_share_weight):
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            each_pred_id = np.argmax(each_predict_prob)
            if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            if outlier(each_target_share_weight[0]):
                counters[-1].Ncorrect += 1.0

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    print(f'test accuracy is {acc_test.item()}')
    exit(0)

# ��δ��붨�����ĸ��Ż������ֱ����ڵ����ĸ���ͬ��������ģ���еĲ������� feature_extractor��classifier��discriminator 
# �� discriminator_separate����Щ�Ż������ǻ�������ݶ��½� (SGD) �㷨ʵ�ֵģ�
# ÿ���Ż�����������ѧϰ�ʡ�Ȩ��˥���Ͷ����ȳ�����������ѧϰ����ʹ����˥�������� (scheduler) ��̬�����ġ�
# ����أ�����ʹ�õ��Ż����� OptimWithSheduler �࣬���̳��� PyTorch �������Ż����� optim.Optimizer�����ڴ˻����������ѧϰ�ʵ�������
# �������������Ǹ���ѵ�������Զ�����ѧϰ�ʣ��Լ���ģ�͵�ѵ��������������õĵ������� inverseDecaySheduler ������
# ����һ������������������ѵ�����������ӣ�ѧϰ�ʻ��𽥼�С��initial_lr * (1 + gamma * step / max_iter) ** (-power)
# ���У�step ��ʾ��ǰ��ѵ��������initial_lr �ǳ�ʼѧϰ�ʣ�gamma �� power �Ƿ����������ĳ�������max_iter �����ѵ��������
# ��ѵ��������ѧϰ����С��ʱ�̡���������������Ǹ��ݵ�ǰѵ�������������һ�����ʵ�ѧϰ�� lr��Ȼ���䴫�ݸ���Ӧ���Ż�����
# �����ĸ��Ż������ǻ��� SGD �㷨ʵ�ֵģ������ÿ���Ż����ж���������ͬ�Ķ��� (momentum) ��Ȩ��˥�� (weight_decay) �ȳ�������
# �Կ���ѵ�����ȶ��Ժͷ����ԡ�

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_domain_discriminator = OptimWithSheduler(
    optim.SGD(domain_discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_reuse_discriminator_s = OptimWithSheduler(
    optim.SGD(reuse_discriminator_s.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_reuse_discriminator_t = OptimWithSheduler(
    optim.SGD(reuse_discriminator_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

# ����global_step��ʼ��Ϊ0�����ڼ�¼ȫ�ֲ���������best_acc��ʼ��Ϊ0�����ڼ�¼��õĲ��Ծ��ȡ�
# total_steps���ڵ���ȫ�ֲ�����ÿ��ѭ������ʱ������ǰ�����ڽ���������ʾ��
# epoch_id��ʼ��Ϊ0�����ڼ�¼��ǰѵ����������

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0

while global_step < args.train.min_step:

    # iters���ڵ���source_train_dl��target_train_dl���ݼ��е�ͼ��ͱ�ǩ��ÿ��ѭ������ʱ������ǰ�����ڽ���������ʾ��
    # ÿ��ѭ�������У���source���ݼ���target���ݼ��е�ͼ��ͱ�ǩ�ֱ�ֵ������im_source��label_source��im_target��label_target��
    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        # save_label_target�������ڵ���ʹ�á� ��label_source��label_targetת��������豸��
        # ��label_targetȫ������Ϊ0������ָʾ��Щͼ��������Ŀ�����ݼ��ġ�

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)
        
        # ��һ������ forward pass����ǰ�򴫲��Ĺ��̡���ͨ������ģ���еķ������������ݽ��д����õ���Ӧ����������
        # ������˵���������Ƚ������Դ��ͼ�� im_source ��Ŀ����ͼ�� im_target �ֱ���������ȡ�� feature_extractor ����������ȡ��
        # �õ������������� fc1_s �� fc1_t��
        # �������������������������ֱ�������� classifier ���з��ࡣ����Դ������������
        # ������������õ� fc1_s, feature_source, fc2_s, predict_prob_source������ fc2_s �Ƿ������������
        # ��ʾԴ��ͼ��ķ�������predict_prob_source ��ʾÿ�����ĸ��ʡ�
        # ���Ƶأ�Ŀ�������������Ҳ�����������õ���Ӧ����������
        # �ڵõ�Դ���Ŀ���������������ʹ�ü����� discriminator �ֱ�������������࣬
        # �õ�Դ���Ŀ������������ domain_prob_discriminator_source �� domain_prob_discriminator_target��
        # ͬʱ������Ҳʹ��������һ�������� discriminator_separate ��Դ���Ŀ����������������з��࣬
        # �õ�Դ���Ŀ������������ domain_prob_discriminator_source_separate �� domain_prob_discriminator_target_separate��
        # ��󣬸��ݵõ��ļ���������Լ�������������ֱ����Դ���Ŀ������ÿ������Ȩ�أ����ڼ���������ʧ������
        # ���У�get_source_share_weight() �������ڼ���Դ��ͼ���Ȩ�أ�get_target_share_weight() �������ڼ���Ŀ����ͼ���Ȩ�أ�
        # normalize_weight() �������ڽ��õ���Ȩ�ؽ��й�һ������

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = domain_discriminator.forward(feature_source)
        domain_prob_discriminator_target = domain_discriminator.forward(feature_target)

        hat_source, max_value_source = perturb(im_source, fc2_s)
        hat_target, max_value_target = perturb(im_target, fc2_t)

        hat_fc1_s = feature_extractor.forward(hat_source.detach())
        hat_fc1_t = feature_extractor.forward(hat_target.detach())

        _, _, hat_fc_2_s, _ = classifier.forward(hat_fc1_s.detach())
        _, _, hat_fc_2_t, _ = classifier.forward(hat_fc1_t.detach())

        source_share_weight = get_source_share_weight(domain_prob_discriminator_source, hat_fc_2_s, max_value_source,
                                                      domain_temperature=1.0, class_temperature=1.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target, hat_fc_2_t, max_value_target,
                                                      domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)
        
        fc2_t, w_avg = pseudo_label_calibration(fc2_t, source_share_weight)

        feature_target_private, feature_target_common = common_private_spilt(target_share_weight, feature_target)
        
        dt_loss = torch.zeros(1, 1).to(output_device)
        ds_loss = torch.zeros(1, 1).to(output_device)
        
        if feature_target_private == torch.Size([]):
            pass
        else:
            fc_target_private, _ = common_private_spilt(target_share_weight, fc2_t)
            reuse_prob_discriminator_private_t = reuse_discriminator_t.forward(feature_target_private)

            target_reuse_weight = get_target_reuse_weight(reuse_prob_discriminator_private_t, fc_target_private)
            tmp = target_reuse_weight * (1 - target_share_weight) * nn.BCELoss(reduction='none')(reuse_prob_discriminator_private_t, 
                                                                 torch.zeros_like(reuse_prob_discriminator_private_t))
            dt_loss += torch.mean(tmp, dim=0, keepdim=True)

        if feature_target_common == torch.Size([]):
            pass
        else:
            reuse_prob_discriminator_common_t1 = reuse_discriminator_t.forward(feature_target_common)
            tmp = nn.BCELoss(reduction='none')(reuse_prob_discriminator_common_t1, 
                                                                 torch.ones_like(reuse_prob_discriminator_common_t1))
            dt_loss += torch.mean(tmp, dim=0, keepdim=True)
            
            reuse_prob_discriminator_common_t2 = reuse_discriminator_t.forward(feature_target_common)
            tmp = nn.BCELoss(reduction='none')(reuse_prob_discriminator_common_t2, 
                                                                 torch.ones_like(reuse_prob_discriminator_common_t2))
            ds_loss += torch.mean(tmp, dim=0, keepdim=True)
        
        feature_source_private, _ = common_private_spilt(source_share_weight, feature_source)
        
        if feature_source_private == torch.Size([]):
            pass
        else:
            fc_source_private, _ = common_private_spilt(source_share_weight, fc2_s)
            reuse_prob_discriminator_private_s = reuse_discriminator_s.forward(feature_source_private)
            
            
            source_reuse_weight = get_source_reuse_weight(reuse_prob_discriminator_private_s, fc_source_private, w_avg, 
                                                      reuse_temperature=1.0, common_temperature = 1.0)
            tmp = source_reuse_weight * nn.BCELoss(reduction='none')(reuse_prob_discriminator_private_s, 
                                                                 torch.zeros_like(reuse_prob_discriminator_private_s))
            ds_loss += torch.mean(tmp, dim=0, keepdim=True)
            
        # ��δ��������ģ�͵���ʧ���������ȣ������˶Կ���ʧ��adv_loss���͵����б����ĶԿ���ʧ��adv_loss_separate����
        # ���ڶԿ���ʧ�����ȸ���Դ��������Ȩ�ؼ���Դ���������ʧ����Σ�����Ŀ����������Ȩ�ؼ���Ŀ�����������ʧ�����������ӡ�
        # ���ڵ����б����ĶԿ���ʧ���ֱ����Դ���Ŀ�����������ʧ��Ȼ����ӡ�
        # �����������㽻������ʧ��ce����ʹ��PyTorch�еĽ�������ʧ��������Ԥ��ֵ����ʵ��ǩ֮��Ľ�������ʧ��
        # ��󣬽��Կ���ʧ�������б����ĶԿ���ʧ�ͽ�������ʧ��Ȩ��͵õ��ܵ���ʧ��loss���������䷴�򴫲���
        # ʹ��OptimizerManager�������ĸ���ͬ���Ż�����optimizer_finetune��optimizer_cls��optimizer_discriminator
        # ��optimizer_discriminator_separate�������ڷ��򴫲�ʱͬʱ�����ĸ��Ż����Ĳ�����

        # ============================= domain loss
        dom_loss = torch.zeros(1, 1).to(output_device)

        tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, 
                                                                 torch.zeros_like(domain_prob_discriminator_source))
        dom_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, 
                                                                 torch.ones_like(domain_prob_discriminator_target))
        dom_loss += torch.mean(tmp, dim=0, keepdim=True)

        # ============================== cross entropy loss
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
        ce = torch.mean(ce, dim=0, keepdim=True)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_domain_discriminator,
                  optimizer_reuse_discriminator_t, optimizer_reuse_discriminator_s]):
            loss = ce + dom_loss + dt_loss + ds_loss
            loss.backward()

        # ���ȣ������е� global_step += 1 ��ʾȫ�ֲ�����һ�����ڼ�¼��ǰѵ���Ľ��ȡ�
        # ��������total_steps.update() ��ʾ�����ۼƲ����ļ����������ڼ���ƽ��ѵ��ʱ�䡣
        # Ȼ��if global_step % args.log.log_interval == 0: �жϵ�ǰȫ�ֲ����Ƿ�Ϊ args.log.log_interval �ı�����
        # ����ǣ����¼һЩѵ��ָ�ꡣ
        # �������Ĵ��붨����һ�� AccuracyCounter ���� counter�����ڼ�¼ѵ�������е�׼ȷ�ʡ�
        # counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
        # ����ǰ���ε���ʵ��ǩ��ģ��Ԥ��ĸ��ʴ��ݸ� counter ������м�¼��
        # Ȼ��ͨ������ counter �����е�׼ȷ�����õ� acc_train��������ת��Ϊ torch.tensor �������ʹ�� logger ��¼����ָ���ֵ��
        # adv_loss��ce��adv_loss_separate �� acc_train��
        # ���У�adv_loss ��ʾ�Կ���ʧ��ce ��ʾ��������ʧ��adv_loss_separate ��ʾ����ĶԿ���ʧ��acc_train ��ʾ��ǰѵ����׼ȷ�ʡ�
        # ���global_step ���� 1��������һ��ѭ����

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('dom_loss', dom_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)
            logger.add_scalar('dt_loss', dt_loss, global_step)
            logger.add_scalar('ds_loss', ds_loss, global_step)
        
        # ��δ�������ÿ��һ���Ĳ�������в��Լ��Ĳ��ԣ�����¼����׼ȷ�ʡ����ԵĹ�����ʹ�����ۼ��� Accumulator 
        # ���ռ����Լ��ϵ�Ԥ�����ͱ�ǩ��Ȼ��ʹ�� AccuracyCounter ����ÿ������׼ȷ�ʡ�����ʵ�ֹ������£�
        # �������ģʽ������ feature_extractor��classifier �� discriminator_separate Ϊ����Ҫ�ݶȸ��µ�ģʽ��
        # ʹ�� Accumulator �ռ����Լ��ϵ� feature��predict_prob��label��domain_prob��before_softmax �� target_share_weight��
        # ���ռ����ı���ת��Ϊ numpy ���飬Ȼ����� updateData ���������ۼ�����
        # ʹ�� outlier �����ж�ÿ�������� target_share_weight �Ƿ�С�� args.test.w_0��
        # ��ʹ�� AccuracyCounter ����ÿ�����Ĳ���׼ȷ�ʣ����ÿ�����Ĳ���׼ȷ�ʴ洢�� acc_tests �б��С�
        # �����������һ�����Ĳ���׼ȷ�ʲ��� nan��������������Ĳ���׼ȷ�ʵ�ƽ��ֵ��������洢�� acc_test �����С�
        # ʹ�� logger ��¼ acc_test �͵�ǰ��ѵ������ global_step��
        # ��� acc_test ����֮ǰ����Ѳ���׼ȷ�� best_acc���򽫵�ǰ��ģ�Ͳ������浽 log_dir/best.pkl �С�
        # ����ǰ��ģ�Ͳ������浽 log_dir/current.pkl �С�

        if global_step % args.test.test_interval == 0:

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, domain_discriminator], train=False) as mgr, \
                 Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax', 'target_share_weight']) as target_accumulator, \
                 torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    feature, __, before_softmax, predict_prob = classifier.forward(feature)
                    
                    before_softmax, _ = pseudo_label_calibration(before_softmax, source_share_weight)
                    predict_prob = before_softmax.softmax(-1)

                    domain_prob = domain_discriminator.forward(__)

                    hat, max_value = perturb(im, before_softmax)

                    hat_feature = feature_extractor.forward(hat)

                    _, _, hat_before_softmax, _ = classifier.forward(hat_feature)

                    target_share_weight = get_target_share_weight(domain_prob, hat_before_softmax,  max_value,
                                                                  domain_temperature=1.0, class_temperature=1.0)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]

            def outlier(each_target_share_weight):
                return each_target_share_weight < args.test.w_0

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

            for (each_predict_prob, each_label, each_target_share_weight) in zip(predict_prob, label,
                                                                                 target_share_weight):
                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    each_pred_id = np.argmax(each_predict_prob)
                    if not outlier(each_target_share_weight[0]) and each_pred_id == each_label:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if outlier(each_target_share_weight[0]):
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)

            logger.add_scalar('acc_test', acc_test, global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'domain_discriminator': domain_discriminator.state_dict() if not isinstance(domain_discriminator, Nonsense) else 1.0,
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)