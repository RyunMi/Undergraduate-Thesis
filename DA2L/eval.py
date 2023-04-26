from data.data_loader import *
from models.net import *
from models.utils import *
from train import feature_extractor,classifier,domain_discriminator,output_device
from tqdm import tqdm

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