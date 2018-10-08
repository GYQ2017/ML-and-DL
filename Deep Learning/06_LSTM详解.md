## LSTM详解

- 时间：2018-07-12
- 摘要：

#### 介绍

---

原始RNN的隐藏层只有一个状态，即h，它对于短期的输入非常敏感。那么，假如我们再增加一个状态，即c，让它来保存长期的状态，就可以解决该问题。

LSTM的关键，就是怎样控制长期状态c 。LSTM的思路是使用三个控制开关：第一个开关，负责控制继续保存长期状态c；第二个开关，负责控制把即时状态输入到长期状态c；第三个开关，负责控制是否把长期状态c作为当前的LSTM的输出

门(gate)实际上就是一层**全连接层**，它的输入是一个向量，输出是一个0到1之间的实数向量。假设W是门的权重向量，b是偏置项，那么门可以表示为: $g(x)=\sigma(Wx+b)$ 。门的使用，就是用门的输出向量按元素乘以我们需要控制的那个向量。因为门的输出是0到1之间的实数向量，那么，当门输出为0时，任何向量与之相乘都会得到0向量，这就相当于啥都不能通过；输出为1时，任何向量与之相乘都不会有任何改变，这就相当于啥都可以通过

#### 前向计算

---

- 遗忘门(forget gate):决定了上一时刻的单元状态$c_{t-1}$有多少保留到当前时刻$c_t$
  $$
  f_t = \sigma(W_f\cdot \left [ h_{t-1},x_t \right ]+b_f)
  $$
  如果输入的维度是$d_x$，隐藏层的维度是$d_h$，单元状态的维度是$d_c$（通常$d_c=d_h$），则遗忘门的权重矩阵$W_f$维度是$d_c\times (d_h+d_x)$ 

- 输入门(input gate):它决定了当前时刻网络的输入$x_t$ 有多少保存到单元状态 $c_t$
  $$
  i_t = \sigma(W_i\cdot \left [ h_{t-1},x_t \right ]+b_i)
  $$
  根据上一次的输出和本次输入来计算当前输入的单元状态:
  $$
  \tilde{c_t}=tanh(W_c\cdot \left[h_{t-1},x_t \right]+b_c)
  $$
  最终当前时刻的单元状态 $c_t$ 的计算：由上一次的单元状态 $c_{t-1}$ 按__元素乘__以遗忘门 $f_t$，再用当前输入的单元状态 $\tilde{c_t}$ 按元素乘以输入门 $i_t$，再将两个积加和。这样，就可以把当前的记忆 $c_t$ 和长期的记忆 $c_{t-1}$ 组合在一起，形成了新的单元状态 $c_t$ 
  $$
  c_t = f_t\circ c_{t-1}+i_t\circ \tilde{c_t}
  $$

- 输出门(output gate):控制单元状态 $c_t$ 有多少输出到 LSTM 的当前输出值 $h_t$
  $$
  o_t=\sigma(W_o\cdot \left [ h_{t-1},x_t \right ]+b_o)
  $$
  LSTM最终的输出，是由输出门和单元状态共同确定的:
  $$
  h_t = o_t\circ tanh(c_t)
  $$




#### 网络训练

---

![06_LSTM](/home/gaoyinquan/MS/Deep Learning/DL_Picture/06_LSTM.png)

LSTM的训练算法仍然是反向传播算法，主要有下面三个步骤：

1. 前向依次计算每个神经元的输出值，即$f_t$、$i_t$、$c_t$、$o_t$、$h_t$五个向量的值
2. 反向计算每个神经元的**误差项** $\delta$值。误差项的反向传播包括两个方向：一个是沿时间的反向传播，即从当前t时刻开始，计算每个时刻的误差项；一个是将误差项向上一层传播
3. 根据相应的误差项，计算每个权重的梯度

##### 误差项沿时间的反向传递

沿时间反向传递误差项，就是要计算出t-1时刻的误差项$\delta_{t-1}$ 
$$
\delta_{t-1}^T=\frac{\partial E}{\partial h_{t-1}}=\frac{\partial E}{\partial h_t}\,\frac{\partial h_t}{\partial h_{t-1}}=\delta_t^T\frac{\partial h_t}{\partial h_{t-1}}
$$
为了求出$\frac{\partial h_t}{\partial h_{t-1}}$ ，我们列出$h_t$的计算公式可知，$o_t、f_t、i_t、\tilde{c_t}$是$h_{t-1}$ 的函数，全函数求导可得：
$$
\delta_t^T\frac{\partial h_t}{\partial h_{t-1}}=\delta_t^T \frac{\partial h_t}{\partial o_t}\, \frac{\partial o_t}{\partial h_{t-1}}+\delta_t^T \frac{\partial h_t}{\partial c_t}\, \frac{\partial c_t}{\partial f_t}\, \frac{\partial f_t}{\partial h_{t-1}}+\delta_t^T \frac{\partial h_t}{\partial c_t}\, \frac{\partial c_t}{\partial i_t}\, \frac{\partial i_t}{\partial h_{t-1}}
$$
最终可以写出将误差项向前传递到任意k时刻的公式：
$$
\delta_k^t=\prod_{j=k}^{t-1}\delta_{o,j}^TW_{oh}+\delta_{f,j}^TW_{fh}+\delta_{i,j}^TW_{ih}+\delta_{\tilde{c_t},j}^TW_{ch}
$$

##### 误差项传递到上一层

我们假设当前为第l层，定义l-1层的误差项是误差函数对l-1层**加权输入**的导数：
$$
\delta_t^{l-1}=\frac{\partial E}{net_t^{l-1}}
$$

##### 权重梯度的计算

对于$W_{fh}$、$W_{ih}$、$W_{ch}$、$W_{oh}$的权重梯度，我们知道它的梯度是各个时刻梯度之和，将各个时刻的梯度加在一起，就能得到最终的梯度：
$$
\frac{\partial E}{\partial W_{fh}}=\sum_{j=1}^t\delta_{f,j}h_{j-1}^T \qquad \frac{\partial E}{\partial b_f}=\sum_{j=1}^t\delta_{f,j} \qquad \frac{\partial E}{\partial W_{fx}}=\delta_{f,t}x_t^T
$$

$$
\frac{\partial E}{\partial W_{ih}}=\sum_{j=1}^t\delta_{i,j}h_{j-1}^T \qquad \frac{\partial E}{\partial b_i}=\sum_{j=1}^t\delta_{i,j} \qquad \frac{\partial E}{\partial W_{ix}}=\delta_{i,t}x_t^T
$$

$$
\frac{\partial E}{\partial W_{ch}}=\sum_{j=1}^t\delta_{\tilde c,j}h_{j-1}^T \qquad \frac{\partial E}{\partial b_c}=\sum_{j=1}^t\delta_{\tilde c,j} \qquad \frac{\partial E}{\partial W_{cx}}=\delta_{\tilde c,t}x_t^T
$$

$$
\frac{\partial E}{\partial W_{oh}}=\sum_{j=1}^t\delta_{o,j}h_{j-1}^T \qquad \frac{\partial E}{\partial b_o}=\sum_{j=1}^t\delta_{o,j} \qquad \frac{\partial E}{\partial W_{ox}}=\delta_{o,t}x_t^T
$$

#### 源码实现

---

##### 反向传播算法的实现

backward方法实现了LSTM的反向传播算法。需要注意的是，与backward相关的内部状态变量是在调用backward方法之后才初始化的

```python
def backward(self,x,delta_h,activator):
	'''实现LSTM训练算法'''
    self.calc_delta(delta_h, activator)
    self.calc_gradient(x)   
```

算法主要分成两个部分，一部分是计算误差项

```python
def calc_delta(self, delta_h, activator):
    # 初始化各个时刻的误差项
	self.delta_h_list = self.init_delta()  # 输出误差项
    self.delta_o_list = self.init_delta()  # 输出门误差项
    self.delta_i_list = self.init_delta()  # 输入门误差项
    self.delta_f_list = self.init_delta()  # 遗忘门误差项
    self.delta_ct_list = self.init_delta() # 即时输出误差项
    # 保存从上一层传递下来的当前时刻的误差项
    self.delta_h_list[-1] = delta_h
    # 迭代计算每个时刻的误差项
    for k in range(self.times, 0, -1):
        self.calc_delta_k(k)
        
def init_delta(self):
	'''初始化误差项'''
    delta_list = []
    for i in range(self.times + 1):
        delta_list.append(np.zeros((self.state_width, 1)))
    return delta_list
    
def calc_delta_k(self, k):
    '''根据k时刻的delta_h，计算k时刻的delta_f、delta_i、delta_o、delta_ct，
    以及k-1时刻的delta_h'''
    # 获得k时刻前向计算的值
    ig = self.i_list[k];	og = self.o_list[k];	fg = self.f_list[k]
    ct = self.ct_list[k];	c = self.c_list[k];		c_prev = self.c_list[k-1]
    tanh_c = self.output_activator.forward(c)
    delta_k = self.delta_h_list[k]
    # 根据式9计算delta_o
    delta_o = (delta_k * tanh_c * self.gate_activator.backward(og))
    delta_f = (delta_k * og * (1 - tanh_c * tanh_c) * c_prev *
               self.gate_activator.backward(fg))
    delta_i = (delta_k * og * (1 - tanh_c * tanh_c) * ct *
               self.gate_activator.backward(ig))
    delta_ct = (delta_k * og * (1 - tanh_c * tanh_c) * ig *
                self.output_activator.backward(ct))
    delta_h_prev = (np.dot(delta_o.transpose(), self.Woh) +
        np.dot(delta_i.transpose(), self.Wih) +
        np.dot(delta_f.transpose(), self.Wfh) +
        np.dot(delta_ct.transpose(), self.Wch)).transpose()
    # 保存全部delta值
    self.delta_h_list[k-1] = delta_h_prev
    self.delta_f_list[k] = delta_f
    self.delta_i_list[k] = delta_i
    self.delta_o_list[k] = delta_o
    self.delta_ct_list[k] = delta_ct
```

另一部分是计算梯度

```python
def calc_gradient(self, x):
    # 初始化遗忘门权重梯度矩阵和偏置项
    self.Wfh_grad, self.Wfx_grad, self.bf_grad = (
        self.init_weight_gradient_mat())
    # 初始化输入门权重梯度矩阵和偏置项
    self.Wih_grad, self.Wix_grad, self.bi_grad = (
        self.init_weight_gradient_mat())
    # 初始化输出门权重梯度矩阵和偏置项
    self.Woh_grad, self.Wox_grad, self.bo_grad = (
        self.init_weight_gradient_mat())
    # 初始化单元状态权重梯度矩阵和偏置项
    self.Wch_grad, self.Wcx_grad, self.bc_grad = (
        self.init_weight_gradient_mat())
    # 计算对上一次输出h的权重梯度
    for t in range(self.times, 0, -1):
        # 计算各个时刻的梯度
        (Wfh_grad, bf_grad,Wih_grad, bi_grad,Woh_grad, bo_grad,Wch_grad, bc_grad) = (self.calc_gradient_t(t))
        # 实际梯度是各时刻梯度之和
        self.Wfh_grad += Wfh_grad;	self.bf_grad += bf_grad
        self.Wih_grad += Wih_grad;	self.bi_grad += bi_grad
        self.Woh_grad += Woh_grad;	self.bo_grad += bo_grad
        self.Wch_grad += Wch_grad;	self.bc_grad += bc_grad    
    # 计算对本次输入x的权重梯度
    xt = x.transpose()
    self.Wfx_grad = np.dot(self.delta_f_list[-1], xt)
    self.Wix_grad = np.dot(self.delta_i_list[-1], xt)
    self.Wox_grad = np.dot(self.delta_o_list[-1], xt)
    self.Wcx_grad = np.dot(self.delta_ct_list[-1], xt)
    
def init_weight_gradient_mat(self):
    '''初始化权重矩阵'''
    Wh_grad = np.zeros((self.state_width,self.state_width))
    Wx_grad = np.zeros((self.state_width,self.input_width))
    b_grad = np.zeros((self.state_width, 1))
    return Wh_grad, Wx_grad, b_grad

def calc_gradient_t(self, t):
    '''计算每个时刻t权重的梯度'''
    h_prev = self.h_list[t-1].transpose()
    Wfh_grad = np.dot(self.delta_f_list[t], h_prev)
    bf_grad = self.delta_f_list[t]
    Wih_grad = np.dot(self.delta_i_list[t], h_prev)
    bi_grad = self.delta_f_list[t]
    Woh_grad = np.dot(self.delta_o_list[t], h_prev)
    bo_grad = self.delta_f_list[t]
    Wch_grad = np.dot(self.delta_ct_list[t], h_prev)
    bc_grad = self.delta_ct_list[t]
    return Wfh_grad, bf_grad, Wih_grad, bi_grad, 
		   Woh_grad, bo_grad, Wch_grad, bc_grad
```

梯度下降算法来更新权重

```python
def update(self):
    '''按照梯度下降，更新权重'''
    self.Wfh -= self.learning_rate * self.Whf_grad
    self.Wfx -= self.learning_rate * self.Whx_grad
    self.bf -= self.learning_rate * self.bf_grad
    self.Wih -= self.learning_rate * self.Whi_grad
    self.Wix -= self.learning_rate * self.Whi_grad
    self.bi -= self.learning_rate * self.bi_grad
    self.Woh -= self.learning_rate * self.Wof_grad
    self.Wox -= self.learning_rate * self.Wox_grad
    self.bo -= self.learning_rate * self.bo_grad
    self.Wch -= self.learning_rate * self.Wcf_grad
    self.Wcx -= self.learning_rate * self.Wcx_grad
    self.bc -= self.learning_rate * self.bc_grad
```

#### 参考文献

1. [零基础入门深度学习(6) - 长短时记忆网络](https://zybuluo.com/hanbingtao/note/581764) 
2. [详解 LSTM](https://www.jianshu.com/p/dcec3f07d3b5) 
