### Attention

- 时间：2018-08-28
- 摘要：

#### 基本Attention

一个attention model通常包含n个参数 $y_1,y_2,...y_n$ 和一个上下文c 。它返回一个z向量，这个向量是 $y_i$ 的加权算术平均值，并且权重是根据 $y_i$ 与给定上下文c 的相关性来选择的，下面表示其原理

![08_attention](/home/gaoyinquan/MS/Deep Learning/DL_Picture/08_attention.png)

首先，我们输入为上下文c 和数据的一部分 $y_i$ ，然后计算 $m_1,m_2,...m_n$ 通过一个tanh层，每个 $m_i$ 的计算都是独立计算出来的，公式为：$m_i = tanh(W_{cm}c+W_{ym}y_i)$ 

然后，我们计算每个weights使用`softmax` ，这里的 $s_i$ 是通常`softmax` 进行归一化后的值$\sum_i s_i =1$，输出z 是所有 $y_i$ 的算术平均，每个权重值表示 $y_i,....y_n$ 和上下文的相关性，公式为：$z=\sum s_iy_i$ 

##### Soft Attention and Hard Attention

上面我们描述的机制称为__Soft Attention__，因为它是一个完全可微的确定性机制，可以插入到现有的系统中，梯度通过注意力机制进行传播，同时它们通过网络的其余部分进行传播。 

Hard Attention是一个随机过程：系统不使用所有隐藏状态作为解码的输入，而是以概率 $s_i$ 对隐藏状态$y_i$ 进行采样。为了进行梯度传播，使用蒙特卡洛方法进行抽样估计梯度。

#### Multi-Head Attention

![Multi-Head Attention](/home/gaoyinquan/MS/Deep Learning/DL_Picture/Multi-Head Attention.png)

Scaled Dot-Product Attention 是我们常用的使用点积进行相似度计算的attention，只是多除了一个 $d_k$ 进行调节，使得内积不至于太大，其中scale为尺度变换，防止输入值过大，mask为掩码保证时间的先后关系，公式如下：
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
Multi-Head Attention 首先Q，K，Q进行一个线性变换，然后输入到Scaled Dot-Product Attention，所谓多头就是做多次。每次Q，K，V进行线性变换的参数W是不一样的。然后将h次的放缩点积attention结果进行拼接，再进行一次线性变换得到的值作为多头attention的结果
$$
head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
MultiHead(Q,K,V) = Concat(head_1,...head_h)W^o
$$

#### Self Attention

传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。

但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。

把encoder端self attention计算的结果加入到decoder端做为K和V，结合decoder自身的输出做为Q，得到decoder端与encoder端 attention之间的依赖关系

![Self Attention](/home/gaoyinquan/MS/Deep Learning/DL_Picture/Self Attention.png)

#### 文本分类中Attention

![text classifier attention](/home/gaoyinquan/MS/Deep Learning/DL_Picture/text classifier attention.png)

公式中的 $W_w$ 和 $b_w$ 为Attention的权重与偏值，$u_w$ 也是需要设置的权重。我们将 $h_t$ 通过一层网络得到 $u_t$ ，然后测量该单词在句子中的重要性，进行`softmax` 归一化
$$
u_t = tanh(W_wh_t+b_w)
$$

$$
\alpha_t = softmax(u_t^Tu_w)
$$

$$
s = \sum_t\alpha_th_t
$$

#### 参考文献

- [Attention Model注意力机制](https://ilewseu.github.io/2018/02/12/Attention%20Model%20%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/) 
- [浅谈NLP中的Attention机制](http://xiaosheng.me/2018/01/13/article121/)  
- [论文：Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) 

