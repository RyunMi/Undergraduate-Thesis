from data.data_loader import *
from models.net import *
from models.utils import *
import datetime#用于处理日期时间
from tqdm import tqdm#用于在命令行界面中显示进度条
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
    #用于判断代码是否在 Jupyter Notebook 中运行，并返回相应的进度条模式。
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

# cudnn.benchmark = True 和 cudnn.deterministic = True 分别启用和禁用了 cudnn 的自动调整机制和随机性，以提高训练速度和结果的稳定性。
# 根据 args.misc.gpus 的值，判断是否使用 GPU 计算。如果 args.misc.gpus 小于 1，说明不使用 GPU 计算，
# 此时将环境变量 CUDA_VISIBLE_DEVICES 设置为空字符串，并将输出设备 output_device 设为 CPU；
# 否则，调用 select_GPUs(args.misc.gpus) 函数选择 args.misc.gpus 个可用的 GPU，并将第一个 GPU 设为输出设备 output_device。
# 使用当前时间生成一个日志目录 log_dir。
# 创建一个 SummaryWriter 实例 logger，用于记录训练过程和结果，并将其保存在 log_dir 目录下。
# 将当前配置 save_config 写入一个 YAML 格式的配置文件，保存在 log_dir 目录下的 config.yaml 文件中。

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

# 定义了一个名为 TotalNet 的神经网络模型，并创建了四个并行化的子模型（feature_extractor、classifier、discriminator 
# 和 discriminator_separate）。
# model_dict 是一个字典，包含两个键值对，对应不同的预训练模型：'resnet50' 对应的是 ResNet50Fc 模型，
# 'resnext101' 对应的是 Resnext101Fc 模型。
# TotalNet 继承了 nn.Module 类，并在其构造函数中定义了四个子模型：feature_extractor、classifier、discriminator 
# 和 discriminator_separate，分别代表特征提取器、分类器、判别器和分离判别器。
# 其中，feature_extractor 使用 args.model.base_model 所指定的预训练模型，classifier 使用自定义的 CLS 类，
# discriminator 和 discriminator_separate 使用自定义的 AdversarialNetwork 类。
# forward 方法定义了前向计算过程，首先通过特征提取器得到特征表示 f，然后通过分类器得到分类结果 y 和辅助变量 _ 和 __，
# 最后通过判别器和分离判别器分别得到对抗损失 d 和 d_0，并将 y、d 和 d_0 作为输出返回。
# totalNet 是 TotalNet 的一个实例。
# feature_extractor、classifier、discriminator 和 discriminator_separate 都是使用 nn.DataParallel 进行并行化处理的子模型。
# 其中，device_ids 参数指定了使用的 GPU 设备 ID，output_device 参数指定了输出结果所在的设备。所有子模型均使用 GPU 进行训练（train(True)）。

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

# 如果命令行参数args.test.test_only为True，那么代码会加载预训练模型的权重文件，然后使用该模型对测试集进行预测，并计算模型的测试准确率。
# 首先，代码会通过torch.load函数加载预训练模型的权重文件，并使用load_state_dict函数将权重文件中的参数值加载到相应的网络模型中，
# 包括feature_extractor、classifier、discriminator和discriminator_separate。
# 然后，代码会创建一个长度为len(source_classes) + 1的列表counters，其中source_classes是源域的类别数。
# 接下来，代码会创建一个TrainingModeManager对象，将三个网络模型(feature_extractor、classifier、discriminator_separate)
# 的train参数都设为False，这样就可以在不更新网络权重的情况下进行测试。然后，代码会创建一个Accumulator对象target_accumulator，
# 用于累积每个样本的feature、predict_prob、label、domain_prob、before_softmax和target_share_weight等数据，这些数据会在下面的计算中用到。
# 接着，代码会通过一个for循环，对测试集中的每张图像进行预测。对于每张图像，代码会将其放到GPU上，并经过feature_extractor
# 和classifier网络的前向传播计算，得到特征向量feature、输出概率predict_prob、真实标签label、领域概率domain_prob和未经softmax的输出
# before_softmax。然后，代码会利用domain_prob和before_softmax计算每个目标样本的target_share_weight，表示样本对源域和目标域的贡献程度。
# 在得到这些数据后，代码会通过globals()函数将这些数据转换为全局变量，并更新target_accumulator。
# 接下来，代码会创建一个名为outlier的函数，该函数接收一个each_target_share_weight参数，
# 如果each_target_share_weight小于阈值args.test.w_0，则返回True；否则返回False。
# 然后，代码会创建一个长度为len(source_classes) + 1的counters列表，其中counters[i]表示源域中类别为i的样本的正确分类数和总数。
# 接着，代码会通过一个for循环，对每个预测输出概率each_predict_prob、真实标签each_label和目标样本的target_share_weight进行遍历。
# 如果each_label在源域中，则将其正确分类数和总数计入相应的counters中。如果each_label不在源域中，则将其正确分类数和总数计入counters[-1]中。
# 最后，代码会计算每个counters的准确率

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

# 这段代码定义了四个优化器，分别用于调整四个不同的神经网络模块中的参数，即 feature_extractor、classifier、discriminator 
# 和 discriminator_separate。这些优化器都是基于随机梯度下降 (SGD) 算法实现的，
# 每个优化器都设置了学习率、权重衰减和动量等超参数，其中学习率是使用逆衰减调度器 (scheduler) 动态调整的。
# 具体地，这里使用的优化器是 OptimWithSheduler 类，它继承自 PyTorch 的内置优化器类 optim.Optimizer，并在此基础上添加了学习率调度器。
# 调度器的作用是根据训练步数自动调整学习率，以加速模型的训练收敛。这里采用的调度器是 inverseDecaySheduler 函数，
# 它是一个反比例函数，随着训练步数的增加，学习率会逐渐减小：initial_lr * (1 + gamma * step / max_iter) ** (-power)
# 其中，step 表示当前的训练步数，initial_lr 是初始学习率，gamma 和 power 是反比例函数的超参数，max_iter 是最大训练步数，
# 即训练过程中学习率最小的时刻。这个函数的作用是根据当前训练步数，计算出一个合适的学习率 lr，然后将其传递给相应的优化器。
# 由于四个优化器都是基于 SGD 算法实现的，因此在每个优化器中都设置了相同的动量 (momentum) 和权重衰减 (weight_decay) 等超参数，
# 以控制训练的稳定性和泛化性。

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

# 变量global_step初始化为0，用于记录全局步数；变量best_acc初始化为0，用于记录最好的测试精度。
# total_steps用于迭代全局步数，每次循环迭代时，将当前步数在进度条中显示。
# epoch_id初始化为0，用于记录当前训练的轮数。

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0

while global_step < args.train.min_step:

    # iters用于迭代source_train_dl和target_train_dl数据集中的图像和标签，每次循环迭代时，将当前轮数在进度条中显示。
    # 每次循环迭代中，将source数据集和target数据集中的图像和标签分别赋值给变量im_source、label_source和im_target、label_target。
    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        # save_label_target变量用于调试使用。 将label_source和label_target转换到输出设备。
        # 将label_target全部设置为0，用于指示这些图像是来自目标数据集的。

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)
        
        # 这一部分是 forward pass，即前向传播的过程。它通过调用模型中的方法对输入数据进行处理，得到相应的输出结果。
        # 具体来说，这里首先将输入的源域图像 im_source 和目标域图像 im_target 分别传入特征提取器 feature_extractor 进行特征提取，
        # 得到两个特征向量 fc1_s 和 fc1_t。
        # 接下来，将这两个特征向量分别传入分类器 classifier 进行分类。对于源域特征向量，
        # 经过分类器后得到 fc1_s, feature_source, fc2_s, predict_prob_source，其中 fc2_s 是分类器的输出，
        # 表示源域图像的分类结果，predict_prob_source 表示每个类别的概率。
        # 类似地，目标域的特征向量也经过分类器得到相应的输出结果。
        # 在得到源域和目标域的特征向量后，使用鉴别器 discriminator 分别对其进行领域分类，
        # 得到源域和目标域的领域概率 domain_prob_discriminator_source 和 domain_prob_discriminator_target。
        # 同时，这里也使用了另外一个鉴别器 discriminator_separate 对源域和目标域的特征向量进行分类，
        # 得到源域和目标域的领域概率 domain_prob_discriminator_source_separate 和 domain_prob_discriminator_target_separate。
        # 最后，根据得到的鉴别器输出以及分类器输出，分别计算源域和目标域中每个类别的权重，用于计算联合损失函数。
        # 其中，get_source_share_weight() 函数用于计算源域图像的权重，get_target_share_weight() 函数用于计算目标域图像的权重，
        # normalize_weight() 函数用于将得到的权重进行归一化处理。

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
            
        # 这段代码计算了模型的损失函数。首先，计算了对抗损失（adv_loss）和单独判别器的对抗损失（adv_loss_separate）。
        # 对于对抗损失，首先根据源域特征的权重计算源域的领域损失，其次，根据目标域特征的权重计算目标域的领域损失，最后将两者相加。
        # 对于单独判别器的对抗损失，分别计算源域和目标域的领域损失，然后相加。
        # 接下来，计算交叉熵损失（ce），使用PyTorch中的交叉熵损失函数计算预测值和真实标签之间的交叉熵损失。
        # 最后，将对抗损失、单独判别器的对抗损失和交叉熵损失加权求和得到总的损失（loss），并将其反向传播。
        # 使用OptimizerManager来管理四个不同的优化器（optimizer_finetune、optimizer_cls、optimizer_discriminator
        # 和optimizer_discriminator_separate），以在反向传播时同时更新四个优化器的参数。

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

        # 首先，代码中的 global_step += 1 表示全局步数加一，用于记录当前训练的进度。
        # 接下来，total_steps.update() 表示更新累计步数的计数器，用于计算平均训练时间。
        # 然后，if global_step % args.log.log_interval == 0: 判断当前全局步数是否为 args.log.log_interval 的倍数，
        # 如果是，则记录一些训练指标。
        # 接下来的代码定义了一个 AccuracyCounter 对象 counter，用于记录训练过程中的准确率。
        # counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
        # 将当前批次的真实标签和模型预测的概率传递给 counter 对象进行记录。
        # 然后，通过计算 counter 对象中的准确率来得到 acc_train，并将其转换为 torch.tensor 对象，最后使用 logger 记录以下指标的值：
        # adv_loss、ce、adv_loss_separate 和 acc_train。
        # 其中，adv_loss 表示对抗损失、ce 表示交叉熵损失、adv_loss_separate 表示分离的对抗损失，acc_train 表示当前训练的准确率。
        # 最后，global_step 增加 1，继续下一轮循环。

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
        
        # 这段代码是在每隔一定的步数后进行测试集的测试，并记录测试准确率。测试的过程中使用了累加器 Accumulator 
        # 来收集测试集上的预测结果和标签，然后使用 AccuracyCounter 计算每个类别的准确率。具体实现过程如下：
        # 进入测试模式，设置 feature_extractor，classifier 和 discriminator_separate 为不需要梯度更新的模式。
        # 使用 Accumulator 收集测试集上的 feature，predict_prob，label，domain_prob，before_softmax 和 target_share_weight，
        # 将收集到的变量转换为 numpy 数组，然后调用 updateData 方法更新累加器。
        # 使用 outlier 函数判断每个样本的 target_share_weight 是否小于 args.test.w_0，
        # 并使用 AccuracyCounter 计算每个类别的测试准确率，最后将每个类别的测试准确率存储到 acc_tests 列表中。
        # 如果存在至少一个类别的测试准确率不是 nan，则计算所有类别的测试准确率的平均值，并将其存储到 acc_test 变量中。
        # 使用 logger 记录 acc_test 和当前的训练步数 global_step。
        # 如果 acc_test 大于之前的最佳测试准确率 best_acc，则将当前的模型参数保存到 log_dir/best.pkl 中。
        # 将当前的模型参数保存到 log_dir/current.pkl 中。

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