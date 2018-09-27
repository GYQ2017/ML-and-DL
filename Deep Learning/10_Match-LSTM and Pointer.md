## Match-LSTM and Answer Pointer

- 时间：2018-09-12
- 摘要：

#### 贡献

1. 我们提出了两种新的用于机器阅读理解的端到端神经网络模型Sequence Model 和Boundary Model，它结合了`match-LSTM`和`Ptr-Net`来处理SQuAD数据集的特殊属性。
2. 在SQuAD v1版本中，模型取得了最好成绩
3. 我们对模型进一步分析提出了一些改进方法，并且也在网上提供我们的代码

#### 结构

- MATCH-LSTM

  `match-lstm`起初用在“文本蕴涵”任务中(即两个文本片段有指向关系，当认为一个片段真实时，可以推断出另一个片段的真实性)。在文本蕴涵中，给出两个句子，其中一个是前提，另一个是假设。为了预测“__前提__”是否包含"__假设__"，`match - LSTM`模型会依次遍历假设的特征。在假设的每个位置，注意力机制被用来获得前提的加权向量表示。然后，这个加权前提与假设的当前特征的向量表示相结合，并被送到一个LSTM中，我们称之为`match - LSTM`。

- Pointer NET

  `Ptr-Net` 不是从固定词汇中选取输出特征, 而是使用注意机制作为一个指针, 从输入序列中选择一个位置作为输出符号。

![10_1](/home/gaoyinquan/MS/Deep Learning/DL_Picture/10_1.png)

- 本文方法

  我们得到了一段文字，我们称之为段落，还有一个与段落相关的问题。段落由矩阵 $P\in\mathbb{R}^{d*P}$ 表示，其中$P$ 是段落长度，$d$ 是word embeddings维度；问题由矩阵 $Q\in\mathbb{R}^{d*Q}$ 表示，其中Q是段落长度。我们的__目标__是从文章中找出一个子序列作为问题的答案。

  Sequence Model：我们将答案表示为整数序列$a = (a_1,a_2,... )$，其中每个$a_i$ 是介于1和P之间的整数，表示段落中的某个位置，非连续；Boundary Model：我们将答案表示为连续序列，预测两个整数$a_s$和$a_e$ ，分别表示段落的开始和结束位置。无论任何模型，我们的训练集是$(P_n,Q_n,a_n)$ 

- 模型概览

  1. LSTM预处理层，使用LSTM预处理文章和问题
  2. Match-LSTM层，将文章和问题相匹配
  3. Answer Pointer层，使用`Ptr-NET`从文章中选择特征作为答案，两种模型区别仅为第三层

#### 模型详解

- LSTM预处理层

  LSTM预处理层的目的是将上下文信息合并到文章和问题中每个特征的表示中。我们使用标准的单向LSTM分别处理文章和问题，如下所示:
  $$
  H^p = \overrightarrow{LSTM}(P), H^q = \overrightarrow{LSTM}(Q)
  $$

- Match-LSTM层

  本文运用 Match-LSTM 模型，将问题作为前提, 段落作为假设。Match-LSTM 顺序遍历段落。在文章的第i位, 它首先使用标准的word-by-word__注意机制__来获得注意权向量$\alpha_i \in \mathbb{R^Q}$ 如下: 
  $$
  \overrightarrow{G_i} = tanh(W^qH^q+(W^ph_i^p+W^r\overrightarrow{h_{i-1}^r}+b^p)\otimes e_Q)
  $$

  $$
  \overrightarrow{\alpha_i} = softmax(w^T\overrightarrow{G_i}+b\otimes e_Q)
  $$

  其中$W^q,W^p,W^r \in\mathbb{R^{l*l}},b^p,w\in\mathbb{R^l}$ 和 $b\in\mathbb{R}$ 等参数需要学习，$\overrightarrow{h}_{i-1}^r \in \mathbb{R^l}$ 是Match-LSTM在第i-1位置的隐藏向量，$(\otimes e_Q)$ 是通过重复左边的向量Q次来产生的矩阵

  $\overrightarrow{\alpha}_{i,j}$ 表明了段落中的第 i 特征与问题中的第j特征之间的匹配程度。接下来，使用$\alpha_i$ 去获得“问题Q”的加权版本，并拼接当前特征的段落形成一个向量$\overrightarrow{z_i}$ 
  $$
  \overrightarrow{z_i} = \begin{bmatrix}
  h_i^p \\
  H^q \overrightarrow{\alpha_i}
  \end{bmatrix}
  $$
  将$\overrightarrow{z_i}$ 送到单向LSTM中，形成Match-LSTM
  $$
  \overrightarrow{h^r_i} = \overrightarrow{LSTM}(\overrightarrow{z_i},\overrightarrow{h^r_{i-1}}),\overrightarrow{h^r_i}\in \mathbb{R^l}
  $$
  我们在相反的方向上进一步构建类似的Match-LSTM，目的是获得一种表示，该表示从两个方向为文章中的每个特征编码上下文。
  $$
  \overleftarrow{G_i} = tanh(W^qH^q+(W^ph_i^p+W^r\overleftarrow{h_{i-1}^r}+b^p)\otimes e_Q)
  $$

  $$
  \overleftarrow{\alpha_i} = softmax(w^T\overleftarrow{G_i}+b\otimes e_Q)
  $$

  $\overrightarrow{H^r} \in \mathbb{R}^{l*P}$ 表示隐藏状态$[\overrightarrow{h_1^r},\overrightarrow{h_2^r},...,\overrightarrow{h_P^r},]$ ，$\overleftarrow{H^r} \in \mathbb{R}^{l*P}$ 表示隐藏状态$[\overleftarrow{h_1^r},\overleftarrow{h_2^r},...,\overleftarrow{h_P^r},]$ 。我们对其进行拼接输入到下一层Pointer层:
  $$
  H^r = \begin{bmatrix}
  \overrightarrow{H^r} \\
  \overleftarrow{H^r}
  \end{bmatrix}
  $$

- Answer Pointer层

  __The Sequence Model__ : 答案由整数序列$a=(a_1,a_2,...)$ 表示原始段落中所选特征的位置。`Ans - Ptr` 层以顺序方式模拟这些整数的生成。因为答案的长度不是固定的，为了在某一点停止生成答案特征，我们允许每个$a_k$取1到P + 1之间的整数值，其中P + 1是表示答案结束的特殊值。一旦$a_k$被设置为P + 1，答案的生成就会停止。

  为了生成第k个答案特征通过$a_k$ ，首先再使用注意力机制获得权重$\beta_k\in \mathbb{R}^{(P+1)}$ ，其中$\beta_{k,j}$ 是从段落中选择第j个特征作为答案第k个特征的概率：
  $$
  F_k = tanh(V\widetilde{H}^r+(W^ah^a_{k-1}+b^a)\otimes e_{P+1})
  $$

  $$
  \beta_k = softmax(v^TF_k+c\otimes e_{P+1})
  $$

  $\widetilde{H}^r \in\mathbb{R}^{2l*(P+1)}=[H^r;0]$ ，其中$V\in\mathbb{R}^{l*2l},W^a\in\mathbb{R}^{l*l},b^a,v\in\mathbb{R}^l$ 和 $c\in\mathbb{R}$ 是要学习的参数，($\otimes e_{P+1}$)和$\otimes e_Q$ 一样，$h_{k-1}^a$是答案LSTM第k-1位置的隐藏单元。定义为：
  $$
  h_k^a = \overrightarrow{LSTM}(\widetilde{H}^r \beta_k^T,h^a_{k-1})
  $$
  我们可以将生成答案序列的概率建模为：
  $$
  p(a|H^r) = \prod_K p(a_k|a_1,a_2,...a_{k-1},H^r)
  $$

  $$
  p(a_k=j|a_1,a_2,...a_{k-1},H^r)=\beta_{k,j}
  $$

  为了训练模型，我们需要最小化loss函数:
  $$
  -\sum_{n=1}^Nlogp(a_n|P_n,Q_n)
  $$
  __The Boundary Model__:  我们只需要预测两个整数$a_s$和$a_e$，与上述模型的主要区别在于，我们不需要将零填充添加到$H^r$中，生成答案的概率简单地建模为:
  $$
  p(a|H^r) = p(a_s|H^r)p(a_e|a_s,H^r)
  $$




#### 实验

我们首先标记化所有的段落, 问题和答案，由此产生的词汇包含117K 不同的单词。我们使用从Glove嵌入的单词来初始化模型，在Glove中找不到的单词被初始化为零向量。在模型的训练过程中, 单词嵌入没有更新。

隐藏层单元设置为150或者300，使用系数为$\beta_1=0.9$和$\beta_2=0.999$ 的`Adamax`优化器，batch为30，没有使用L2正则化

我们的边界模型优于序列模型，精确匹配分数为61.1 %，F1分数为71.2 %。观察到大多数答案是相对较小的跨度，我们简单地将最大预测跨度限制为不超过15个特征，并进行跨度搜索实验，这导致F1在开发数据上提高了1.5 %。通过在双向预处理 LSTM 中加入双向`Ans-Ptr`, 可使 F1 得到1.2% 的改进。最后，我们通过简单地计算从5个边界模型收集的边界概率的乘积，然后用不超过15个标记搜索最可能的跨度来探索集成方法。如表中所示，这种集成方法达到了最佳性能。

#### 进一步分析

1. 更长的答案越难预测
2. “why”问题更难得到答案
3. 注意力机制是否有利于答案的定位，“why”问题不清楚那个单词可以对齐为“why”，其他类型问题得到了较好的对齐

#### 参考文献

- [MurtyShikhar 源码实现](https://github.com/MurtyShikhar/Question-Answering) 
- [InnerPeace-Wu 源码实现](https://github.com/InnerPeace-Wu/reading_comprehension-cs224n) 更简洁

