import yaml
import easydict
from os.path import join

# ������һ����Ϊ "Dataset" ���ࡣ�����ĸ����ԣ����� "path"�� "prefix"�� "domains" �� "files"��
# "path" �����ݼ����ڵ�·����"prefix" �����ݼ���ǰ׺���ƣ�"domains" �����ݼ��а���������б�"files" �����ݼ��а����������ļ����б�
# ����Ĺ��캯���У����� "path" �� "prefix" ����Ϊ����Ĳ��������� "domains" ��ÿ�����ǰ׺����Ϊ "prefix"��
# ���⣬�������ļ��б� "files" �е�ÿ���ļ���·������Ϊ "path" + �ļ�����

class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)

import argparse

# ������һ�����������в����Ķ����������нű�ʱ�������ж�ȡ�������ڴ����У�argparse.ArgumentParser() 
# ������һ�� ArgumentParser ���ʵ�������ݸ����캯���� description ������һ���ַ����������˽������Ĺ��ܣ�
# ���Ի��ڶԿ�ѧϰ������Ӧ��Domain Adaptation Based on Adversarial Learning�����д���ʵ�֡�
# formatter_class ��������ָ������������Ϣ�ĸ�ʽ��
# �������ʹ����Ĭ�ϵ� argparse.ArgumentDefaultsHelpFormatter �࣬������������в����İ�����Ϣ����ʾĬ��ֵ��

parser = argparse.ArgumentParser(description='Code for *Domain Adaptation Based on Adversarial Learning*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ͨ�������н����� argparse �����һ����Ϊ config �������в�����
# �������������Ϊ�ַ�����ȱʡֵΪ 'config.yaml'��������һ��������Ϣ /path/to/config/file��
# ֮��ʹ�� parser.parse_args() �������������в���������洢�� args �����С�
# Ȼ�������ļ���·���� args ����ȡ��������ȡ�����ļ�������洢�� args �С�
# ��󣬽� args ����ת��Ϊ�ɷ������ԵĶ��� EasyDict ������洢�� args �С�
# Ҫָ��ĳ��yaml���ã�python main.py --config config/officehome-train-config.yaml

parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()
config_file = args.config
args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

# ���ݲ��� args.data.dataset.name ��ֵ��ѡ���ض������ݼ������統��ֵΪ office ʱ������һ�� Dataset ���� dataset��
# ���а����� amazon��dslr �� webcam ������������ݣ���Щ�����ļ���·���� args.data.dataset.root_path �£�
# �ļ����ֱ�Ϊ amazon.txt��dslr.txt �� webcam.txt�� 
# officehome��domainnet��visda2017����
# ��󣬸��� args.data.dataset.source �� args.data.dataset.target ��ֵ��ȡԴ���Ŀ��������ƺ��ļ�·����
# ���洢�� source_domain_name��target_domain_name��source_file �� target_file �����С�

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