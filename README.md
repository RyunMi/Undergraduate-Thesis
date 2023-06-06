# XTU-thesis

Xiangtan University Undergraduate Thesis

## Paper-Reading

Some .md notes about Semi-Supervised Learning, Generative Adversarial Networks and Domain Adaptation's paper.

## Thesis-Template

A LaTeX template about Xiangtan University's undergraduate thesis.

You can use your LaTeX compile to run the *xtuthesis.tex* to complete your abstrct(by modifing the *abstract.tex*), body, reference, acknowledgements and appendix. 

And you should modify the *cover-template.doc* to complete your cover and some other preface part.

## $DA^2L$ 

Code Release for "Domain Adaptation Based on Adversarial Learning".

### Enviroment

Ubuntu 20.04 LTS

Nvidia GeForce RTX 3090

Python 3.7

PyTorch 1.7.1

`pip install -r requirements.txt`

### Usage

- download datasets(Office-31, Office-Home, VisDA2017, DomainNet et al.) and pretrained models(such as ResNet50)

- write your tran & test config file

- train:

  `python main.py --config config/(train_config_name).yaml`

- test:

  `python main.py --config config/(test_config_name).yaml`

- monitor (tensorboard required):

  `tensorboard --logdir log/(dataset)/(time)/`

### Note

Inspired by [youkaichao](https://github.com/thuml/Universal-Domain-Adaptation) and [zhuohuangai](https://github.com/zhuohuangai/cafa-1).
