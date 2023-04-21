import yaml
import easydict
from os.path import join

# 定义了一个名为 "Dataset" 的类。它有四个属性，包括 "path"， "prefix"， "domains" 和 "files"。
# "path" 是数据集所在的路径，"prefix" 是数据集的前缀名称，"domains" 是数据集中包含的域的列表，"files" 是数据集中包含的所有文件的列表。
# 在类的构造函数中，它将 "path" 和 "prefix" 设置为传入的参数，并将 "domains" 中每个域的前缀设置为 "prefix"。
# 此外，它还将文件列表 "files" 中的每个文件的路径设置为 "path" + 文件名。

class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)

import argparse

# 定义了一个解析命令行参数的对象，用于运行脚本时从命令行读取参数。在代码中，argparse.ArgumentParser() 
# 构造了一个 ArgumentParser 类的实例，传递给构造函数的 description 参数是一个字符串，描述了解析器的功能，
# 即对基于对抗学习的域适应（Domain Adaptation Based on Adversarial Learning）进行代码实现。
# formatter_class 参数用于指定参数帮助信息的格式。
# 在这里，它使用了默认的 argparse.ArgumentDefaultsHelpFormatter 类，该类会在命令行参数的帮助信息中显示默认值。

parser = argparse.ArgumentParser(description='Code for *Domain Adaptation Based on Adversarial Learning*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 通过命令行解析器 argparse 添加了一个名为 config 的命令行参数。
# 这个参数的类型为字符串，缺省值为 'config.yaml'，并且有一个帮助信息 /path/to/config/file。
# 之后，使用 parser.parse_args() 函数解析命令行参数，将其存储在 args 变量中。
# 然后将配置文件的路径从 args 中提取出来，读取配置文件并将其存储在 args 中。
# 最后，将 args 变量转换为可访问属性的对象 EasyDict 并将其存储在 args 中。
# 要指定某个yaml配置：python main.py --config config/officehome-train-config.yaml

parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()
config_file = args.config
args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

# 根据参数 args.data.dataset.name 的值来选择特定的数据集，比如当其值为 office 时，创建一个 Dataset 对象 dataset，
# 其中包含了 amazon、dslr 和 webcam 三个领域的数据，这些数据文件的路径在 args.data.dataset.root_path 下，
# 文件名分别为 amazon.txt、dslr.txt 和 webcam.txt。 
# officehome，domainnet，visda2017类似
# 最后，根据 args.data.dataset.source 和 args.data.dataset.target 的值获取源域和目标域的名称和文件路径，
# 并存储在 source_domain_name、target_domain_name、source_file 和 target_file 变量中。

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['amazon', 'dslr', 'webcam'],
        files=[
            'amazon.txt',
            'dslr.txt',
            'webcam.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'officehome':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['Art', 'Clipart', 'Product', 'Real_World'],
        files=[
            'Art.txt',
            'Clipart.txt',
            'Product.txt',
            'Real_World.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'domainnet':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        files=[
            'clipart_train.txt',
            'infograph_train.txt',
            'painting_train.txt',
            'quickdraw_train.txt',
            'real_train.txt',
            'sketch_train.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'visda2017':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['train', 'validation'],
        files=[
            'train_list.txt',
            'validation_list.txt'
        ],
        prefix=args.data.dataset.root_path)
    dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]