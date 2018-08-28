## RNN详解

- 时间：2018-07-12
- 摘要：主要讲述了RNN和传统神经网络的不同，以及RNN的前向传播、后向传播和部分实现

#### 概述

在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。循环神经网络(Recurrent Neural Networks)思想就是使用序列信息，具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，__即隐藏层之间的节点不再无连接而是有连接的__，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

以单层的循环神经网络为例，介绍RNN如何进行__前向传播__：

![Screenshot from 2018-07-13 09-15-45](/home/gaoyinquan/MS/Deep Learning/DL_Picture/05_RNN.png)

$s_0=\vec{0}，s_1=g_1(w_{ss}*s_0+w_{sx}*x_1+b_s)， y_1 = g_2(w_{sy}*s_1+b_y)​$ 。其中​$g_1​$ 通常为​$tanh/ReLu​$ ，​$g_2​$ 通常为​$sigmoid(二分类)/softmax(多分类)​$ 。

一般情况下可简化为：$s_t = g_1(w_s[s_{t-1},x_t]+b_s)$ ，例如假设$s_o=\vec{100},w_{ss}=100*100,x_1=\vec{10000}$ ，$w_{sx}=100*10000$ ，那么$w_s=[w_{ss},w_{sx}]=100*10100, [s_{t-1},x_t]=[{s_{t-1} \choose x_t}] = 10100*100$ 。 

#### 反向传播BPTT

我们的目标是计算误差关于参数U、V和W的梯度，然后使用梯度下降法学习出好的参数。为了计算这些梯度，我们需要使用微分的链式法则 。我们以第3步的loss $E_3$为例：$\frac{\partial E_3}{\partial V} = \frac{\partial E_3}{\partial \hat{y_3}} * \frac{\partial \hat{y_3}}{\partial V} = (\hat{y_3}-y_3)\bigotimes{s_3}$ ，可看出$\frac{\partial E_3}{\partial V}$ 仅仅依赖当前时刻的值，如$\hat{y_3}、y_3、s_3$ 等。但是计算$\frac{\partial E_3}{\partial W}$和$\frac{\partial E_3}{\partial U}$却有所不同，我们写出链式法则，$\frac{\partial E_3}{\partial W} = \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial s_3} \frac{\partial s_3}{\partial W}$ ，其中$s_3 = tanh(U*x_3+W*s_2)$ 依赖于$s_2$ ，而$s_2$ 依赖于$s_1$ 。所以我们需要再次应用链式法则，$\frac{\partial E_3}{\partial W} = \sum_{k=0}^3 \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial s_3} \frac{\partial s_3}{\partial s_k} \frac{\partial s_k}{\partial W}$ 。我们将每时刻对梯度的贡献相加，也就是说，我们需要从时刻 t = 3 通过网络的所有路径到时刻 t = 0 来反向传播梯度：

![05_RNN02](/home/gaoyinquan/MS/Deep Learning/DL_Picture/05_RNN02.png)

请留意，这与我们在深度前馈神经网络中使用的标准反向传播算法完全相同。主要的差异就是我们将每时刻 W 的梯度相加。在传统的神经网络中，我们在层之间并没有共享参数，所以我们不需要相加。

#### 源码

假如我们有一个包含m个单词的句子，语言模型允许我们预测在已知数据集中观察到的句子的概率。

- 初始化RNN

```python
class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
```

- 前向传播

```python
def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s 
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. 
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]
```

- 计算损失

```python
def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
```

- 使用SGD计算反向传播BPTT

```python
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o 
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
```

#### 参考文献

- [Andrew Ng 的课程DeepLearning.ai](http://mooc.study.163.com/university/deeplearning_ai#/c) 
- [BPTT推导](https://www.cnblogs.com/zhbzz2007/p/6339346.html) 
- [python实现RNN](http://www.cnblogs.com/zhbzz2007/p/6291652.html) 
