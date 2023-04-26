from data.data_loader import *
from models.net import *
from models.utils import *
from train import feature_extractor,classifier,domain_discriminator,output_device
from tqdm import tqdm

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