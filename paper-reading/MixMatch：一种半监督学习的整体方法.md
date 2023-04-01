# MixMatch：一种半监督学习的整体方法

## 引用方式

### GB/T 7714

<span style="background-color: rgb(255, 255, 255)">Berthelot D, Carlini N, Goodfellow I, et al. Mixmatch: A holistic approach to semi-supervised learning[J]. Advances in neural information processing systems, 2019, 32.</span>

### BibTex

    @article{berthelot2019mixmatch,
      title={Mixmatch: A holistic approach to semi-supervised learning},
      author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin A},
      journal={Advances in neural information processing systems},
      volume={32},
      year={2019}
    }

## 开源代码

[YU1ut/MixMatch-pytorch: Code for "MixMatch - A Holistic Approach to Semi-Supervised Learning" (github.com)](https://github.com/YU1ut/MixMatch-pytorch)

## 主要思想

半监督学习被证明是一种利用未标记数据减轻对大型标记数据集的依赖的强大范式。

本文统一了当前主要的半监督学习方法，开发了一种新算法MixMatch，通过猜测经过数据增强的未标记示例的低熵标签和使用MixUp混合有标记和未标记数据来工作。

本文展示了MixMatch在许多数据集和标记数据量方面都取得了大幅领先的最新结果。例如，在 CIFAR-10 上使用 250 个标签，将错误率降低了4倍(从38%到11%)，在 STL-10 上降低了2倍。

本文还展示了MixMatch如何帮助实现更好的精度隐私权衡的差异性隐私。最后进行了一项消融研究，以分离MixMatch成功的哪些成分是最重要的。

## 主要内容

### 引论

最近许多半监督学习方法都添加了一种在未标记数据上计算的损失项，以促使模型更好地推广到未见过的数据。

在许多最近的研究中，这个损失项可以分为三类：熵最小化--它鼓励模型对未标记数据输出自信的预测。一致性正则化--鼓励模型在其输入被扰动时生成相同的输出分布；通用正则化--鼓励模型很好地泛化，避免过度拟合训练数据。

MixMatch是一种半监督学习 SSL 算法，其引入了单一损失并优雅地统一了这些主流半监督学习方法。与以往的方法不同，MixMatch一次性针对所有特性进行目标设置。简而言之，MixMatch 引入了一个统一的未标记数据损失项，可以在保持一致性的情况下无缝地降低熵，并与传统的正则化技术兼容。

### 相关工作

给出一个通用模型 $p_{model}(y|x;\theta)$ ，该模型使用参数 $\theta$ 为输入 $x$ 生成类标签 $y$ 的分布。

#### 一致性正则化

在监督学习中常用的正则化技术是数据增强，该技术应用于输入转换，假定不会影响类语义。例如在图像分类中，通常会弹性变形或加入噪声到输入图像中，这可以大大改变图像的像素内容而不改变其标签。粗略地说，这可以通过生成接近无限的新的修改数据的流来人为地扩大训练集的大小。

一致性正则化将数据增强应用于半监督学习，利用了一个分类器应该在扩充之后输出相同的类分布的思想。

$\Pi$ 模型和均值教师模型采用此正则化，这些方法的缺点是它们使用特定于领域的数据增强策略。VAT解决了这个问题，而不是直接更改输入的概率分布，该方法计算一种加性扰动，并应用于原输入上，该扰动能够最大化更改输出的类别分布。

MixMatch使用标准的数据增强技术(随机水平翻转和裁剪)来实现一种一致性正则化形式。

#### 熵最小化

许多半监督学习方法的共同假设是分类器的决策边界不应通过边缘数据分布的高密度区域。实现这一点的一种方法是要求分类器在未标记的数据上输出低熵预测。

通过简单地添加一个损失项可显式实现，该项可最小化未标记数据 $x$ 的 $p_{model}(y|x;\theta)$ 的熵。这种形式的熵最小化与VAT相结合可获得更强的结果。

“伪标签”方法通过在未标记数据上构建高置信度预测的硬标签，并将这些作为标准交叉熵损失的训练目标，隐式地执行熵最小化操作。

MixMatch还通过在未标记数据的目标分布上使用“锐化”功能隐含地实现了熵最小化。

#### 传统正则化

正则化指的是对模型施加约束的一般方法，使其更难记忆训练数据，因此希望它对未见过的数据有更好的泛化能力。一种普遍的正则化技术是添加一个损失项，其是惩罚模型参数的 $L_2$ 范数，这可以看作是在权重值上强制实施零均值协方差高斯先验。

使用简单的梯度下降时，这个损失项相当于指数衰减权重值趋近于零。而使用Adam作为梯度优化器时，使用显式的“权重衰减”而不是 $L_2$ 损失项。

MixUp正则化项使用输入和标签的凸组合来训练模型。MixUp可以被认为是鼓励模型在样本之间具有严格线性行为的一种方法，因为它要求模型在合并两个输入的凸组合时的输出接近于每个单独输入的输出的凸组合。

本文在MixMatch中使用MixUp既作为正则化方法(应用于有标签数据点)，又作为半监督学习方法(应用于无标签数据点)。

### MixMatch

MixMatch是一种“整体”方法，它融合了之前讨论的SSL主导范式中的思想和成分。

给定一批包含标签示例 $\mathcal{X}$ 和相应的独热目标(代表 $L$ 个可能标签中的一个)以及大小相同的未标记示例 $\mathcal{U}$ 。

MixMatch生成了一批处理过的带标签样本 $\mathcal{X}'$ 和一批带有“猜测”标签的处理过的未标记样本 $\mathcal{U}'$ 。将 $\mathcal{U}'$ 和 $\mathcal{X}'$ 用于分开计算带标签和未标记损失项。

更正式地，半监督学习的联合损失 $L$ 被计算为：

![\<img alt="" data-attachment-key="LIUY2F2S" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22VWZIARFL%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B210.762%2C116.32%2C402.365%2C209.867%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%223%22%7D%7D" width="319" height="156" src="attachments/LIUY2F2S.png" ztype="zimage">](attachments/LIUY2F2S.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%223%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 3</a></span>)</span>

其中 $H(p,q)$ 是分布 $p$ 和 $q$ 之间的交叉熵，而 $T,K,\alpha,\lambda_{\mathcal{U}}$ 是超参数。

完整的MixMatch算法如下图所示：

![\<img alt="" data-attachment-key="WI26RILQ" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%229R73H5TZ%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%224%22%2C%22position%22%3A%7B%22pageIndex%22%3A3%2C%22rects%22%3A%5B%5B104.254%2C488.818%2C508.873%2C724.376%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%7D" width="674" height="392" src="attachments/WI26RILQ.png" ztype="zimage">](attachments/WI26RILQ.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 4</a></span>)</span>

标签猜测过程如下图所示：

![\<img alt="" data-attachment-key="WF9RRYN5" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22JMTL8RYT%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B111.58011049723756%2C641.4232061396647%2C504.36499999999984%2C719.7546978523718%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%222%22%7D%7D" width="655" height="131" src="attachments/WF9RRYN5.png" ztype="zimage">](attachments/WF9RRYN5.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%222%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 2</a></span>)</span>

#### 数据增强

如前所述，减轻标记数据不足的常见方法是使用数据增强。数据增强引入了一个函数 $Augment(x)$ ，该函数以随机的方式将输入数据点 $x$ 进行转换，使其标签保持不变，不同的Augment应用将产生不同的(随机)输出。在有标签和无标签数据上都使用数据增强。

对于标记数据批次 $\mathcal{X}$ 中的每个 $x_b$ ，本文生成一个变换版本 $\hat{x}\_b=Augment(x_b)$ (算法第3行)。对于未标记数据批次  $\mathcal{U}$ 中的每个 $u_b$ ，本文生成 $K$ 个增强 $\hat{u}\_{b,k}=Augment(u_b)$ ，其中 $k\in(1,\cdots,K)$ (算法第5行)。这些分离的增强被用于生成每个 $u_b$ 的“猜测标签” $q_b$ 。

#### 标签猜测

对于 $\mathcal{U}$ 中的每个未标记示例，MixMatch使用模型的预测生成示例标签的“猜测”，这个猜测后来被用于无监督损失项。为此，计算 $u_b$ 的 $K$ 个增强的模型预测类分布的平均值(算法第7行)：

![\<img alt="" data-attachment-key="8G3QB6NJ" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22IU4ED5E4%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%224%22%2C%22position%22%3A%7B%22pageIndex%22%3A3%2C%22rects%22%3A%5B%5B240.066%2C246.497%2C371.37%2C281.436%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%7D" width="219" height="58" src="attachments/8G3QB6NJ.png" ztype="zimage">](attachments/8G3QB6NJ.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 4</a></span>)</span>

在一致性正则化方法中，使用数据增强来获取未标记示例的人工目标是常见的。

#### 锐化

在生成标签猜测时，本文执行一个额外的步骤，灵感来自熵最小化在半监督学习中的成功。

鉴于增强的平均预测值 $\overline{q}_b$ ，本文用锐化函数来降低标签分布的熵。

实践中，对于锐化函数，本文使用常见的方法来调整这个分类分布的“温度”，该操作被定义为：

![\<img alt="" data-attachment-key="A4DYWBJB" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22V2XRWWZL%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%224%22%2C%22position%22%3A%7B%22pageIndex%22%3A3%2C%22rects%22%3A%5B%5B238.376%2C97.38564911731709%2C372.4972375690607%2C130.6342679018474%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%7D" width="224" height="56" src="attachments/A4DYWBJB.png" ztype="zimage">](attachments/A4DYWBJB.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%224%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 4</a></span>)</span>

其中 $p$ 是某个输入的分类分布(在MixMatch中， $p$ 是增强的平均类别预测，见算法第8行)。

$T$ 是超参数，当 $T\rightarrow 0$ 时， $Sharpen(p,T)$ 的输出将趋近于Dirac(“one-hot”)分布。

因为将 $q_b=Sharpen(\hat{q_b},T)$ 作为模型对 $u_b$ 增强的预测目标，降低温度会鼓励模型产生更低熵的预测。

#### Mixup

要使用MixUp进行半监督学习，本文将其应用于带有标签的示例和具有标签猜测的未标记示例。与过去使用MixUp进行 SSL的工作不同，本文将有标签的示例与无标签的示例进行“混合”，反之亦然，发现这样可以提高性能。

在联合损失函数中，为标记和未标记的数据使用分离的损失项。在最初提出的形式下使用MixUp会引起问题。相反对于一对具有其对应的独热标签的两个示例 $(x_1,p_1),(x_2,p_2)$ ，本文定义了一个稍微修改的MixUp，通过计算 $(x',p')$ 来实现( $\alpha$ 是超参数)：

![\<img alt="" data-attachment-key="KHL5SSVE" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22AR6R5672%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B255.845%2C498.398%2C356.718%2C560.387%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%7D" width="168" height="103" src="attachments/KHL5SSVE.png" ztype="zimage">](attachments/KHL5SSVE.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 5</a></span>)</span>

最初的提议MixUp可以被看作是省略了上图第二个等式(即设 $\lambda'=\lambda$ ）。

要应用MixUp，首先将所有增强的标记示例及其标签收集起来(算法第10行)：

![\<img alt="" data-attachment-key="YBCW9LFP" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22BPZZBIRL%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B241.757%2C449.37%2C369.116%2C466.276%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%7D" width="212" height="28" src="attachments/YBCW9LFP.png" ztype="zimage">](attachments/YBCW9LFP.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 5</a></span>)</span>

将所有未标记示例的所有增强以及猜测的标签合并在一起(算法第11行)：

![\<img alt="" data-attachment-key="7P9GBLZY" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22TIQV6ZGF%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B205.691%2C417.249%2C404.055%2C435.845%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%7D" width="331" height="31" src="attachments/7P9GBLZY.png" ztype="zimage">](attachments/7P9GBLZY.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 5</a></span>)</span>

然后将这些集合合并并洗牌以形成 $\mathcal{W}$ ，作为MixUp的数据源(算法第12行)。

对于 $\hat{\mathcal{X}}$ 中的每个第 $i$ 个示例-标签对，计算 $MixUp(\hat{\mathcal{X}_i},\mathcal{W}_i)$ ，并将结果添加到集合 $\mathcal{X}'$ 中(算法第13行)。注意，由于对MixUp进行了轻微修改，因此 $\mathcal{X}'$ 中的元素在插值方面比 $\mathcal{W}$ 中相应的插补更接近原始标记数据点。

类似地计算 $\mathcal{U}_i'=MixUp(\hat{\mathcal{U}_i},\mathcal{W}_{i+|\hat{\mathcal{X}}|})$ ，其中 $i\in(1,\dots,|\hat{\mathcal{U}}|)$ ，有意使用没有在构建 $\mathcal{X}'$ 中使用的 $\mathcal{W}$ 的剩余部分(算法第14行)。

总之，MixMatch 将 $X'$ 转化为 $\mathcal{X}'$ ，这是一组应用了数据增强和MixUp(可能混合了未标记的示例)的标记示例。同样地， $\mathcal{U}$ 被转化为 $\mathcal{U}'$ ，其中包含了每个未标记示例的多个增强以及相应的标签猜测。

#### 损失函数

鉴于使用MixMatch生成的处理批次 $\mathcal{X}'$ 和 $\mathcal{U}'$ ，使用前述标准半监督损失：

![\<img alt="" data-attachment-key="CUBNMBSL" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22TRQW8K5H%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B222.5966850828729%2C118.12376378517796%2C402.3646408839778%2C194.76464776307847%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%223%22%7D%7D" width="300" height="128" src="attachments/CUBNMBSL.png" ztype="zimage">](attachments/CUBNMBSL.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%223%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 3</a></span>)</span>

上图第三个方程将来自 $\mathcal{X}'$ 的标签和模型预测之间的典型交叉熵损失与来自 $\mathcal{U}'$ 的预测和猜测标签的平方 $L_2$ 损失相结合。

上图第二个方程中的平方 $L_2$ 损失对应于多类Brier分数，与交叉熵不同，它是有界的，对完全错误的预测不太敏感。因此，它经常被用作半监督学习中未标记数据的预测损失以及预测不确定性的度量。

注意，第二个方程中的猜测标签 $q$ 是模型参数的函数；然而当使用这种损失函数的形式时，不会通过猜测标签传播梯度。

#### 超参数

由于MixMatch结合了多个机制来利用未标记的数据，因此引入了各种超参数：锐化温度 $T$ 、未标记数据增强数量 $K$ 、Beta中的 $\alpha$ 参数以及无监督损失权重 $\lambda_{\mathcal{U}}$ 。

总的来说，具有许多超参数的半监督学习方法在实践中应用可能会有问题，因为很难使用小验证集进行交叉验证。

然而在实践中发现，大多数MixMatch的超参数都可以固定并不需要在每个实验或每个数据集的基础上进行调整。

## 结论及改进方向

### 实验

除非另有说明，在所有实验中，使用“Wide ResNet-28”模型，并且与原论文在某些技巧有所区别。

使用的数据集：CIFAR-10和CIFAR-100，SVHN和STL-10。STL-10是专门设计用于半监督学习的数据集，其中包含5000张有标注的图片和100000张无标注的图片，这些无标注的图片来自于与有标注数据略有不同的分布。

#### 结果

CIFAR-10的结果如下图：

![\<img alt="" data-attachment-key="Z5P256A2" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22YT4JTYHG%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%227%22%2C%22position%22%3A%7B%22pageIndex%22%3A6%2C%22rects%22%3A%5B%5B103.691%2C613.923%2C299.801%2C722.122%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%7D" width="327" height="180" src="attachments/Z5P256A2.png" ztype="zimage">](attachments/Z5P256A2.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 7</a></span>)</span>

发现MixMatch比所有其他方法表现都好得多。

对于更大的模型，结果如下图：

![\<img alt="" data-attachment-key="4SLVD3TD" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22UQZMFD7Z%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%227%22%2C%22position%22%3A%7B%22pageIndex%22%3A6%2C%22rects%22%3A%5B%5B104.818%2C440.24088569767565%2C307.6906077348066%2C504.5969999999999%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%7D" width="338" height="107" src="attachments/4SLVD3TD.png" ztype="zimage">](attachments/4SLVD3TD.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 7</a></span>)</span>

总的来说，MixMatch与\[2]中的最佳结果相匹配或表现更好。

SVHN的结果如下图：

![\<img alt="" data-attachment-key="N85JGMX4" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22MJD8RNWB%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%227%22%2C%22position%22%3A%7B%22pageIndex%22%3A6%2C%22rects%22%3A%5B%5B312.762%2C615.05%2C506.055%2C723.812%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%7D" width="322" height="181" src="attachments/N85JGMX4.png" ztype="zimage">](attachments/N85JGMX4.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 7</a></span>)</span>

发现MixMatch的性能在所有标记数据量上都相对稳定(并且比所有其他方法都要好)。

对 SVHN 和 SVHN+Extra 进行 MixMatch 的错误率比较结果如下图：

![\<img alt="" data-attachment-key="5NGEMDMM" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22HHWSCEAK%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%228%22%2C%22position%22%3A%7B%22pageIndex%22%3A7%2C%22rects%22%3A%5B%5B107.072%2C673.094%2C504.928%2C724.939%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%228%22%7D%7D" width="663" height="86" src="attachments/5NGEMDMM.png" ztype="zimage">](attachments/5NGEMDMM.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%228%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 8</a></span>)</span>

发现，在这两组训练集上，MixMatch几乎立即接近了完全监督的表现。

STL-10错误率结果如下图：

![\<img alt="" data-attachment-key="BPJM7YMB" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22AUSIIJIP%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%227%22%2C%22position%22%3A%7B%22pageIndex%22%3A6%2C%22rects%22%3A%5B%5B320.652%2C431.901%2C506.055%2C513.05%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%7D" width="309" height="135" src="attachments/BPJM7YMB.png" ztype="zimage">](attachments/BPJM7YMB.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%227%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 7</a></span>)</span>

#### 消融实验

1.使用K个数据增强的平均类别分布或使用单个数据增强的类别分布(即设K设置为1)。

2.去除温度锐化(即设T = 1)。

3.使用指数移动平均值(EMA)的模型参数来产生猜测的标签，就像均值教师所做的那样。

4.仅对标记示例执行MixUp；仅对未标记示例执行MixUp；或者不在标记和未标记示例之间混合执行MixUp。

5.通过使用插值一致性训练，可以将其视为这种消融研究的特殊情况，其中仅使用未标记的mixup，不应用锐化，并使用EMA参数进行标签猜测。

![\<img alt="" data-attachment-key="23NHC5QP" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDW5BELBT%22%2C%22annotationKey%22%3A%22XHMQAMYL%22%2C%22color%22%3A%22%23ffd400%22%2C%22pageLabel%22%3A%229%22%2C%22position%22%3A%7B%22pageIndex%22%3A8%2C%22rects%22%3A%5B%5B124.541%2C595.89%2C487.459%2C722.685%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%229%22%7D%7D" width="605" height="211" src="attachments/23NHC5QP.png" ztype="zimage">](attachments/23NHC5QP.png)\
<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F8273795%2Fitems%2FDG73HB5E%22%5D%2C%22locator%22%3A%229%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/DG73HB5E">Berthelot 等, 2019, p. 9</a></span>)</span>

#### MixMatch，隐私保护学习，泛化性

以隐私方式学习是评估方法泛化能力的一种绝佳方式。事实上，保护训练数据的隐私相当于证明模型没有过拟合：如果向学习算法添加、修改或删除任何一个训练样本都不会导致学习的模型参数在统计上有显著差异，那么该算法就被称为是差分隐私的。因此，以差分隐私方式进行学习在实践中其实是一种正则化形式。

对训练数据的每一次访问都可能泄漏私人信息。这些敏感信息通常编码在输入和其标签之间的对应关系中。因此，从私人训练数据进行深度学习的方法在计算模型参数更新时尽可能访问尽可能少的标记私人训练点是有益的。半监督学习是这种情况下的天然选择。本文展示了MixMatch在具有差分隐私的学习中明显优于最新技术水平。

本文使用PATE框架来进行带有隐私的学习。

学生从公共的未标记数据中以半监督的方式进行训练，部分数据由一组能够访问私人标记训练数据的教师标记。学生需要的标签越少，以达到固定的准确性，提供的隐私保证就越强。教师使用有噪声的投票机制来回应学生的标签查询，当他们无法达成足够强的共识时，他们可以选择不提供标签。

因此，MixMatch提高PATE的性能也说明MixMatch从每个类别的少量典型实例中改进了泛化能力。

### 未来工作

在未来的工作中，有兴趣将半监督学习文献中的其他想法融入到混合方法中，并继续探索哪些成分能产生有效的算法。
