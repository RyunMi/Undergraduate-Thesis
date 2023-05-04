from data.data_loader import *
from models.net import *
from models.utils import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(args.misc.gpus)
    device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d-%H_%M_%S')

log_dir = f'{args.log.root_dir}{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'resnext101':Resnext101Fc
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.domain_discriminator = AdversarialNetwork(256)
        #self.domain_discriminator_separate = AdversarialNetwork(256)
        self.reuse_discriminator_s = AdversarialNetwork(256)
        self.reuse_discriminator_t = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.domain_discriminator(_)
        #d_0 = self.domain_discriminator_separate(_)
        hat_d_s = self.reuse_discriminator_s(_)
        hat_d_t = self.reuse_discriminator_t(_)
        return y, d, hat_d_t, hat_d_s #d_0#, hat_d_s, hat_d_t

totalNet = TotalNet()
totalNet.to(device)

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=device).train(True)
domain_discriminator = nn.DataParallel(totalNet.domain_discriminator, device_ids=gpu_ids, output_device=device).train(True)
reuse_discriminator_s = nn.DataParallel(totalNet.reuse_discriminator_s, device_ids=gpu_ids, output_device=device).train(True)
reuse_discriminator_t = nn.DataParallel(totalNet.reuse_discriminator_t, device_ids=gpu_ids, output_device=device).train(True)

# =================== evaluation
if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    domain_discriminator.load_state_dict(data['domain_discriminator'])
    # w_avg.load_state_dict(data['w_avg'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier, domain_discriminator], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax',
                         'target_share_weight']) as target_accumulator:#, \
                #torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(device)
            label = label.to(device)

            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature)

            # predict_prob, _ = pseudo_label_calibration(predict_prob, w_avg)
                    
            domain_prob = domain_discriminator.forward(__)

            pred_shift = perturb(im, feature_extractor, classifier, class_temperature=1.0)

            target_share_weight = get_target_share_weight(domain_prob, pred_shift, domain_temperature=1.0)

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

# =================== optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr /10.0, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_domain_discriminator = OptimWithSheduler(
    optim.SGD(domain_discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,
               momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_reuse_discriminator_t = OptimWithSheduler(
    optim.SGD(reuse_discriminator_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_reuse_discriminator_s = OptimWithSheduler(
    optim.SGD(reuse_discriminator_s.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0
total_epoch = min(len(source_train_dl), len(target_train_dl))
source_share_weight_epoch = torch.zeros(total_epoch * args.data.dataloader.batch_size, 1).to(device)
label_source_epoch = torch.zeros(total_epoch * args.data.dataloader.batch_size).to(device)
w_avg = torch.zeros(len(source_classes)).to(device)

# =================== train
while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=total_epoch)
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(device)
        label_target = label_target.to(device)
        label_target = torch.zeros_like(label_target)

        # ========================= forward pass
        im_source = im_source.to(device)
        im_target = im_target.to(device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)
         
        # Output of Domain Discriminator
        domain_prob_discriminator_source = domain_discriminator.forward(feature_source)
        domain_prob_discriminator_target = domain_discriminator.forward(feature_target)

        # domain_prob_discriminator_source_separate = domain_discriminator_separate.forward(feature_source.detach())
        # domain_prob_discriminator_target_separate = domain_discriminator_separate.forward(feature_target.detach())

        # Adversarial perturbation
        pred_shift_source = perturb(im_source, feature_extractor, classifier, class_temperature=10.0)
        pred_shift_target = perturb(im_target, feature_extractor, classifier, class_temperature=1.0)

        # w_s and w_t
        source_share_weight = get_source_share_weight(domain_prob_discriminator_source, pred_shift_source,
                                                    domain_temperature=1.0)
        source_share_weight_norm = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target, pred_shift_target,
                                                    domain_temperature=1.0)
        target_share_weight_norm = normalize_weight(target_share_weight)

        # Pseudo Label Calibration
        source_share_weight_epoch[(global_step % total_epoch) * args.data.dataloader.batch_size : 
                        (global_step % total_epoch + 1) * args.data.dataloader.batch_size] \
            = source_share_weight.clone()
        label_source_epoch[(global_step % total_epoch) * args.data.dataloader.batch_size : 
                        (global_step % total_epoch + 1) * args.data.dataloader.batch_size] \
            = label_source.clone()
        
        if epoch_id == 1:
            w_avg = compute_avg_weight(source_share_weight_epoch[0 : (global_step % total_epoch + 1) * args.data.dataloader.batch_size],
        label_source_epoch[0 : (global_step % total_epoch + 1) * args.data.dataloader.batch_size], w_avg)
            _, w_avg = pseudo_label_calibration(fc2_t, w_avg, pslab_temperature = 1.0)
        else:
            w_avg = compute_avg_weight(source_share_weight_epoch, label_source_epoch, w_avg)
            _, w_avg = pseudo_label_calibration(fc2_t, w_avg, pslab_temperature = 1.0)

        # # Reuse Detect and Reuse Loss
        feature_target_private, feature_target_common = common_private_spilt(target_share_weight, feature_target)
        
        dt_loss = torch.zeros(1, 1).to(device)
        ds_loss = torch.zeros(1, 1).to(device)
        
        if min(feature_target_private.shape) == 0:
            pass
        else:
            fc_target_private, _ = common_private_spilt(target_share_weight, fc2_t)
            # target_share_weight_private, _ = common_private_spilt(target_share_weight, target_share_weight_norm)
            reuse_prob_discriminator_private_t = reuse_discriminator_t.forward(feature_target_private)

            target_reuse_weight = get_target_reuse_weight(reuse_prob_discriminator_private_t, fc_target_private)
            # * (1 / (1 + target_share_weight_private)).view(-1)
            tmp = target_reuse_weight  * nn.BCELoss(reduction='none')\
                (reuse_prob_discriminator_private_t, torch.zeros_like(reuse_prob_discriminator_private_t)).view(-1)
            dt_loss += torch.mean(tmp, dim=0, keepdim=True)

        if min(feature_target_common.shape) == 0:
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
        
        if min(feature_source_private.shape) == 0:
            pass
        else:
            fc_source_private, _ = common_private_spilt(source_share_weight, fc2_s)
            reuse_prob_discriminator_private_s = reuse_discriminator_s.forward(feature_source_private)
            
            source_reuse_weight = get_source_reuse_weight(reuse_prob_discriminator_private_s, fc_source_private, w_avg, 
                                                    reuse_temperature=1.0, common_temperature = 10.0)
            tmp = source_reuse_weight * nn.BCELoss(reduction='none')(reuse_prob_discriminator_private_s, 
                                                                torch.zeros_like(reuse_prob_discriminator_private_s)).view(-1)
            ds_loss += torch.mean(tmp, dim=0, keepdim=True)

        # ============================= domain loss
        dom_loss = torch.zeros(1, 1).to(device)
        # dom_loss_separate = torch.zeros(1, 1).to(device)

        tmp = source_share_weight_norm * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, 
                                                                torch.zeros_like(domain_prob_discriminator_source))
        dom_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight_norm * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, 
                                                                torch.ones_like(domain_prob_discriminator_target))
        dom_loss += torch.mean(tmp, dim=0, keepdim=True)

        # ============================== cross entropy loss
        ce = nn.CrossEntropyLoss(reduction='none')(predict_prob_source, label_source)
        ce = torch.mean(ce, dim=0, keepdim=True)

        dom_coef = 1.0 * math.exp(-5 * (1 - min(global_step / 8000, 1))**2)
        reuse_coef = 1.0 * math.exp(-5 * (1 - min(global_step / 16000, 1))**2)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_domain_discriminator,optimizer_reuse_discriminator_t, optimizer_reuse_discriminator_s]):
                #optimizer_domain_discriminator_separate,]): 
            loss = ce + dom_coef * dom_loss + reuse_coef * dt_loss + ds_loss#+ dom_loss_separate
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(device)
            logger.add_scalar('dom_loss', dom_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('dt_loss', dt_loss, global_step)
            logger.add_scalar('ds_loss', ds_loss, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        # =================== validation
        if global_step % (args.test.test_interval) == 0:

            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier, domain_discriminator], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label', 'domain_prob', 'before_softmax','target_share_weight']) as target_accumulator:#, \
                #torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(device)
                    label = label.to(device)

                    feature = feature_extractor.forward(im)
                    feature, __, before_softmax, predict_prob = classifier.forward(feature)
                    
                    pred_shift_target = perturb(im, feature_extractor, classifier, class_temperature=1.0)

                    # predict_prob, _ = pseudo_label_calibration(predict_prob, w_avg)
                    
                    domain_prob = domain_discriminator.forward(__)

                    target_share_weight = get_target_share_weight(domain_prob, pred_shift_target,
                                                                domain_temperature=1.0)

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
            print(f'test accuracy is {acc_test.item()}')

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'domain_discriminator': domain_discriminator.state_dict() if not isinstance(domain_discriminator, Nonsense) else 1.0,
                # 'w_avg': w_avg.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)