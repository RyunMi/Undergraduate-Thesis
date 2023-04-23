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
# 首先读取配置文件中的参数，并根据 n_share、n_source_private 和 n_total 计算出相应的类别信息，
# 包括共有的类别、源域私有的类别和目标域私有的类别。
a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

# 定义训练和测试时的图像预处理操作，包括将图像大小调整为256x256，然后从中随机裁剪224x224大小的图像，
# 再随机水平翻转图像，最后将图像转换为张量。

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

# 代码定义了四个数据集变量，分别为source_train_ds、source_test_ds、target_train_ds、target_test_ds。
# 这些数据集是基于FileListDataset类构建的，该类实现了一个通用的文件列表数据集，文件名包括类别信息，如"dog/001.jpg"。
# 在构建数据集时，需要指定数据列表的路径(list_path)、路径前缀(path_prefix)、转换函数(transform)和数据集筛选器(filter)。
# 在这里，数据列表的路径分别为source_file和target_file，路径前缀是args.data.dataset.source和args.data.dataset.target，
# 转换函数是train_transform和test_transform，数据集筛选器是lambda函数，用于只保留特定的类别。

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=train_transform, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file,path_prefix=dataset.prefixes[args.data.dataset.source],
                            transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                            transform=test_transform, filter=(lambda x: x in target_classes))

# 用于计算类别权重，并使用这些权重构建采样器。在这里，计算类别频率和类别权重的方式是首先使用Counter类计算数据集中每个类别的数量，
# 然后根据类别频率计算类别权重。如果参数args.data.dataloader.class_balance为True，则使用类别频率计算类别权重，否则将类别权重设置为1.0。
# 最后，根据类别权重计算样本的权重。

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]

# 使用WeightedRandomSampler构建采样器sampler。该采样器从数据集中随机采样样本，并以指定的权重对样本进行加权，用于训练深度学习模型。

sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

# 定义了四个DataLoader对象，分别用于载入源域和目标域的训练和测试数据集。
# 其中source_train_ds、source_test_ds、target_train_ds和target_test_ds分别为之前定义的四个数据集对象，
# 分别用于载入源域的训练和测试数据集以及目标域的训练和测试数据集。
# 而source_train_dl、source_test_dl、target_train_dl和target_test_dl则是对应的四个DataLoader对象，
# 用于设置不同的数据集载入方式，包括batch_size（每个batch的样本数）、sampler（采样方式）、shuffle（是否打乱样本顺序）、
# num_workers（使用的线程数）和drop_last（是否丢弃最后一个batch，如果该batch的样本数小于batch_size）。
# 其中，source_train_dl和target_train_dl的shuffle都被设置为True，即在载入训练数据时随机打乱样本的顺序。

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, drop_last=False)