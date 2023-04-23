from config.config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
# ���ȶ�ȡ�����ļ��еĲ����������� n_share��n_source_private �� n_total �������Ӧ�������Ϣ��
# �������е����Դ��˽�е�����Ŀ����˽�е����
a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

# ����ѵ���Ͳ���ʱ��ͼ��Ԥ���������������ͼ���С����Ϊ256x256��Ȼ���������ü�224x224��С��ͼ��
# �����ˮƽ��תͼ�����ͼ��ת��Ϊ������

train_transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor()
])

# ���붨�����ĸ����ݼ��������ֱ�Ϊsource_train_ds��source_test_ds��target_train_ds��target_test_ds��
# ��Щ���ݼ��ǻ���FileListDataset�๹���ģ�����ʵ����һ��ͨ�õ��ļ��б����ݼ����ļ������������Ϣ����"dog/001.jpg"��
# �ڹ������ݼ�ʱ����Ҫָ�������б��·��(list_path)��·��ǰ׺(path_prefix)��ת������(transform)�����ݼ�ɸѡ��(filter)��
# ����������б��·���ֱ�Ϊsource_file��target_file��·��ǰ׺��args.data.dataset.source��args.data.dataset.target��
# ת��������train_transform��test_transform�����ݼ�ɸѡ����lambda����������ֻ�����ض������

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

# ���ڼ������Ȩ�أ���ʹ����ЩȨ�ع�����������������������Ƶ�ʺ����Ȩ�صķ�ʽ������ʹ��Counter��������ݼ���ÿ������������
# Ȼ��������Ƶ�ʼ������Ȩ�ء��������args.data.dataloader.class_balanceΪTrue����ʹ�����Ƶ�ʼ������Ȩ�أ��������Ȩ������Ϊ1.0��
# ��󣬸������Ȩ�ؼ���������Ȩ�ء�

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]

# ʹ��WeightedRandomSampler����������sampler���ò����������ݼ��������������������ָ����Ȩ�ض��������м�Ȩ������ѵ�����ѧϰģ�͡�

sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

# �������ĸ�DataLoader���󣬷ֱ���������Դ���Ŀ�����ѵ���Ͳ������ݼ���
# ����source_train_ds��source_test_ds��target_train_ds��target_test_ds�ֱ�Ϊ֮ǰ������ĸ����ݼ�����
# �ֱ���������Դ���ѵ���Ͳ������ݼ��Լ�Ŀ�����ѵ���Ͳ������ݼ���
# ��source_train_dl��source_test_dl��target_train_dl��target_test_dl���Ƕ�Ӧ���ĸ�DataLoader����
# �������ò�ͬ�����ݼ����뷽ʽ������batch_size��ÿ��batch������������sampler��������ʽ����shuffle���Ƿ��������˳�򣩡�
# num_workers��ʹ�õ��߳�������drop_last���Ƿ������һ��batch�������batch��������С��batch_size����
# ���У�source_train_dl��target_train_dl��shuffle��������ΪTrue����������ѵ������ʱ�������������˳��

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)