from data.data_loader import *
from models.net import *
from models.utils import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummarWriter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

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
feature_extractor.to(output_device)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
classifier.to(output_device)
domain_discriminator = nn.DataParallel(totalNet.domain_discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
domain_discriminator.to(output_device)
reuse_discriminator_s = nn.DataParallel(totalNet.reuse_discriminator_s, device_ids=gpu_ids, output_device=output_device).train(True)
reuse_discriminator_s.to(output_device)
reuse_discriminator_t = nn.DataParallel(totalNet.reuse_discriminator_t, device_ids=gpu_ids, output_device=output_device).train(True)
reuse_discriminator_t.to(output_device)

# =================== evaluation
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

# =================== optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, 
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
optimizer_reuse_discriminator_s = OptimWithSheduler(
    optim.SGD(reuse_discriminator_s.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_reuse_discriminator_t = OptimWithSheduler(
    optim.SGD(reuse_discriminator_t.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, 
              momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0
total_epoch = min(len(source_train_dl), len(target_train_dl))
stable_softmax = torch.zeros((total_epoch * args.data.dataloader.batch_size, len(source_classes))).to(output_device)
source_share_weight_epoch = torch.zeros(total_epoch * args.data.dataloader.batch_size).to(output_device)
w_avg = torch.zeros(len(source_classes)).to(output_device)

# =================== train
while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=total_epoch)
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # ========================= forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)
        predict_prob_source = TempScale(fc2_s, args.train.temp).softmax(1)
        predict_prob_target = TempScale(fc2_t, args.train.temp).softmax(1)

        # Temporal Ensembling
        if epoch_id != 1:
            predict_prob_target = predict_prob_target * (1 - args.train.alpha)
            predict_prob_target = predict_prob_target.add(stable_softmax[(global_step % total_epoch) 
                * args.data.dataloader.batch_size : ((global_step + 1) % total_epoch) * args.data.dataloader.batch_size],
                  alpha = args.train.alpha)
            
        stable_softmax[(global_step % total_epoch) * args.data.dataloader.batch_size : 
                           ((global_step + 1) % total_epoch) * args.data.dataloader.batch_size] \
            = predict_prob_target.clone()

        # Output of Domain Discriminator
        domain_prob_discriminator_source = domain_discriminator.forward(feature_source)
        domain_prob_discriminator_target = domain_discriminator.forward(feature_target)

        # Adversarial perturbation
        hat_source, max_value_source = perturb(im_source, feature_extractor, classifier)
        hat_target, max_value_target = perturb(im_target, feature_extractor, classifier)

        hat_fc1_s = feature_extractor.forward(hat_source.detach())
        hat_fc1_t = feature_extractor.forward(hat_target.detach())

        _, _, hat_fc_2_s, _ = classifier.forward(hat_fc1_s.detach())
        _, _, hat_fc_2_t, _ = classifier.forward(hat_fc1_t.detach())

        # w_s and w_t
        source_share_weight = get_source_share_weight(domain_prob_discriminator_source, hat_fc_2_s, max_value_source,
                                                      domain_temperature=1.0, class_temperature=1.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target, hat_fc_2_t, max_value_target,
                                                      domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)
        
        # Pseudo Label Calibration
        source_share_weight_epoch[(global_step % total_epoch) * args.data.dataloader.batch_size : 
                           ((global_step + 1) % total_epoch) * args.data.dataloader.batch_size] \
            = source_share_weight
        w_avg = compute_avg_weight(source_share_weight_epoch, label_source, w_avg)

        if epoch_id == 1:
            _, w_avg = pseudo_label_calibration(predict_prob_target, w_avg)
        else:
            predict_prob_target, w_avg = pseudo_label_calibration(predict_prob_target, w_avg)

        # Reuse Detect and Reuse Loss
        feature_target_private, feature_target_common = common_private_spilt(target_share_weight, feature_target)
        
        dt_loss = torch.zeros(1, 1).to(output_device)
        ds_loss = torch.zeros(1, 1).to(output_device)
        
        if feature_target_private == torch.Size([]):
            pass
        else:
            fc_target_private, _ = common_private_spilt(target_share_weight, predict_prob_target)
            target_share_weight_private, _ = common_private_spilt(target_share_weight, target_share_weight)
            reuse_prob_discriminator_private_t = reuse_discriminator_t.forward(feature_target_private)

            target_reuse_weight = get_target_reuse_weight(reuse_prob_discriminator_private_t, fc_target_private)
            tmp = target_reuse_weight * (1 - target_share_weight_private) * nn.BCELoss(reduction='none')\
                (reuse_prob_discriminator_private_t, torch.zeros_like(reuse_prob_discriminator_private_t))
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

        # =================== validation
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