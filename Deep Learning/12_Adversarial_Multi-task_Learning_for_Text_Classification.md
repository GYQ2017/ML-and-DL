## Adversarial Multi-task Learning for Text Classification

- 时间：2018-10-06
- 摘要：在大多数现有方法中，提取的共享特征容易被特定任务特征或其他任务带来的噪声污染。我们提出了一个对抗式多任务学习框架，减轻了共享和私有特征空间之间的相互干扰

#### 介绍

---

大多数多任务学习的现有工作，试图将不同任务的特征划分为私有和共享。如图1-a所示，`shard-private`模型为任何任务引入了两个特征空间：一个用于存储任务相关特征，另一个用于捕获共享特征。这个框架的主要限制是共享特征空间可能包含一些不必要的特定任务特征，而一些共享特征也可能在私有空间中混合，存在特征冗余

为了解决这个问题，我们提出了`对抗性多任务`框架，其中通过引入正交性约束，共享和私有特征空间本质上是不相交的。具体来说，我们设计了一个通用的共享-私有学习框架来模拟文本序列。

为了防止共享和私有特征空间相互干扰，我们引入了两种策略:对抗训练和正交约束。对抗性训练用于确保共享特征空间简单地包含公共和任务不变信息，而正交性约束用于消除私有和共享空间中的冗余特征。

本文的贡献如下：

- 模型以更精确的方式划分特定任务的空间和共享空间，而不是粗略地共享参数
- 我们将原来的二进制对抗性训练扩展到多类，这不仅使多项任务能够联合训练，还允许我们利用未标记的数据
- 我们可以将多个任务之间的共享知识浓缩成现成的神经层，这可以很容易地转移到新的任务中

#### 多任务学习分类

---

多任务学习的目标是利用这些相关任务之间的相关性，通过并行的学习任务来改进分类效果。为了促进这一点，我们对本文中使用的符号给出了一些解释。$D_k$为任务k的具有$N_k$个样本的数据集
$$
D_k=(x_i^k,y_i^k)_{i=1}^{N_k}
$$

##### 2.1 两种方式

![12_Multi](/home/gaoyinquan/MS/Deep Learning/DL_Picture/12_Multi.png)

1. __Fully-Shared 模型__：使用单个共享LSTM层来提取所有任务的特征。例如，给定两个任务m和n，它认为任务m的特征可以被任务n完全共享，反之亦然
2. __Share-Private 模型__：为每个任务引入了两个特征空间：一个用于存储任务相关特征，另一个用于捕获任务不变特征。因此，我们可以看到每个任务都被分配了一个私有的LSTM层和共享的LSTM层。对于任务k的任何句子，我们可以计算其共享表示$s_t^k$ 和私有表示$h_t^k$ :

$$
s_t^k=LSTM(x_t,s_{t-1}^k,\theta_s)
$$

$$
h_t^k=LSTM(x_t,s_{t-1}^m,\theta_k)
$$

##### 2.2 特定任务输出层

对于任务k中的一个句子，其特征$h^{(k)}$由多任务体系结构发出，最终被输入到相应的特定任务的`softmax`层进行分类或其他任务，公式如下，其中$\alpha_k$是每个任务k的权重
$$
L_{Task}=\sum_{k=1}^K\alpha_kL(\hat{y}^k,y^k)
$$

#### 结合对抗性训练

---

![12_Adversarial](/home/gaoyinquan/MS/Deep Learning/DL_Picture/12_Adversarial.png)

尽管__share-Private__模型将特征空间分为共享和私有空间，但不能保证共享的特征不能存在于私有特征空间中，反之亦然。因此，一些有用的可共享特征可能会被忽略，共享特征空间也容易受到某些特定任务信息的污染

因此，一个好的共享特征空间应该包含更多的公共信息，而不包含特定任务的信息。为了解决这个问题，提出了对抗性训练多任务学习框架

##### 3.1 对抗性网络

GAN学习`生成网络G`和`判别模型D`，其中G从生成器分布$p_G(x)$生成样本，D学习确定样本是来自$p_G(x)$还是$P_{data}(x)$ ，通过以下进行优化
$$
\varnothing =\underset{G}{min}\ \underset{D}{max}\left(E_{x\sim P_{data}}\left [ logD(x) \right ]+E_{z\sim p(z)}\left [ log(1-D(G(z)) \right ]\right)
$$

##### 3.2 任务对抗性损失

我们提出了一个用于多任务学习的__adversarial shared-private 模型__，其中共享的网络层正朝着一个可学习的多层感知器进行对抗性的工作，阻止它对任务类型做出准确的预测。这种对抗性训练鼓励共享空间更加纯净，并确保共享表示不受特定任务特征的污染

- 任务鉴别：鉴别器用于将句子的共享表示映射到概率分布中，评估编码的句子来自哪种任务

$$
D(s_T^k,\theta_D)=softmax(b+Us_T^k),(U和b是可学习参数)
$$

- 对抗损失：与大多数现有的多任务学习算法不同，我们增加了一个额外的任务对抗性损失$L_{Adv}$，以防止特定任务的特征蔓延到共享空间。任务对抗性损失用于训练模型以产生共享特征，这样分类器就不能基于这些特征可靠地预测任务。初始对抗网络的损失是有限的，因为它只能在二进制情况下使用。为了克服这一点，我们将其扩展到多类形式，这使得我们的模型可以与多项任务一起训练:
  $$
  L_{Adv}=\underset{\theta_s}{min}\left ( \lambda \underset{\theta_D}{max}(\sum_{k=1}^K \sum_{i=1}^{N_k}d_i^klog\left [ D(E(x^k)) \right ]) \right )
  $$
  其中$d_i^k$表示表示当前任务的真实标签，这里有一个最小-最大优化。基本思想是，给定一个句子，共享LSTM生成一个表示来误导任务鉴别器。同时，鉴别器会尽力对任务类型进行正确分类。在训练之后，共享特征提取器和任务鉴别器到达一个既不能改进又不能区分所有任务的点

- 半监督学习多任务学习：我们注意到$L_{Adv}$ 只需要输入句子x，不需要对应的标签y，这使得半监督学习有可能与我们的模型结合起来。最终，在这个半监督多任务学习框架中，我们的模型不仅可以利用相关任务的数据，还可以利用大量未标记的语料库

##### 3.3 正交约束

我们注意到上述模型有一个潜在的缺点，即任务不变特征可以出现在共享空间和私有空间中。因此，我们引入正交性约束，惩罚冗余的潜在表示，并鼓励共享和私有提取器对输入的不同方面进行编码。在探索许多可选方法后，发现下面的损失是最佳的：
$$
L_{diff}=\sum_{k=1}^K\left \| S^{k^T}H^k \right \|_F^2
$$
其中，$S^k$和$H^k$ 是两个矩阵，它们的行是共享特征提取器的输出$E_s(,;\theta_s)$和特定任务提取的输入句子的$E_k(,;\theta_k)$ 

##### 3.4 合并

最终Loss为：$L=L_{Task}+\lambda L_{Adv}+\gamma L_{Diff}$ ，其中$\lambda$和$\gamma$ 是超参数

#### 实验

##### 4.1 超参数

word embedding：Glove；        mini-batch：16；    其他参数通过从均匀分布[0.1,0.1]中随机采样来初始化；

对于每一项任务，我们都采用超参数，这些超参数通过对初始学习率的组合进行小网格搜索，在开发集上获得最佳性能。最终确定$lr=0.01,\lambda=0.05,\gamma=0.01$ 

##### 4.2 共享知识迁移

借助于对抗性学习，共享特征提取器$E_s$可以生成更纯的任务不变表示，这可以被认为是现成的知识，然后用于新任务。我们研究了两种迁移机制，即单通道和双通道

##### 4.3 可视化

