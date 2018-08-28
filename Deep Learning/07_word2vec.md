### 词向量

---

- 时间：2018-08-17
- 摘要：主要介绍了词表示的发展，从one-hot、特征化表示，到传统DNN模型，最后到基于霍夫曼树的DNN模型和基于负采样的DNN模型

#### 词表示发展

1. one-hot

   假设共有10000个词，其中Man表示为[0,0,0,...1,0,0,...0] ("1"所在位置为5391)，简记为O$_{5391}$ 

   缺点：

   - 向量稀疏
   - 词之间太独立，无联系

2. 特征化表示(Dristributed representation)

   它的思路是通过训练，将每个词都映射到一个较短的词向量上来。所有的这些词向量就构成了向量空间，进而可以用普通的统计学的方法来研究词与词之间的关系

   假设有10000词、300个特征，据此可得到__300×10000__的向量，其中Man可表示为e$_{5391}$ ，e$_{Man}$-e$_{Woman}$=e$_{King}$-e$_w$(近似等于)，其中目标为最大化e$_w$，一般通过余弦相似度计算相似度，即sim(e$_w$,e$_{King}$-e$_{Man}$-e$_{Woman}$)，也可以使用欧氏距离等

   | 特征 | Man  | Woman | King  | Queen | Apple | Orange | ...  |
   | :--: | :--: | :---: | :---: | :---: | :---: | :----: | :--: |
   | 性别 |  -1  |   1   | -0.95 | 0.97  | 0.00  |  0.01  |      |
   | 年龄 | 0.03 | 0.02  | 0.70  | 0.69  | 0.03  | -0.02  |      |
   | 食品 | 0.09 | 0.01  | 0.02  | 0.01  | 0.95  |  0.97  |      |
   | 高贵 | 0.01 | 0.02  | 0.93  | 0.95  | 0.01  |  0.00  |      |
   | ...  |      |       |       |       |       |        |      |

3. DNN 模型

   在`word2vec`出现之前，已有用神经网络来用训练词向量进而处理词与词之间的关系了。采用的方法一般是一个三层的神经网络结构（当然也可以多层），分为输入层，隐藏层和输出层

- CBOW模型

  CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。比如我们的上下文大小取值为4，特定的这个词是"Learning"，也就是我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是训练样本特定词对应的softmax概率最大）

  CBOW神经网络模型输入层有8个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数可自己指定。通过反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量

- Skip-Gram模型

  Skip-Gram模型输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量。还是上面的例子，我们的上下文大小取值为4， 特定的这个词"Learning"是我们的输入，而这8个上下文词是我们的输出。

  Skip-Gram神经网络模型输入层有1个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数可以自己指定。通过反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。当有新需求时，要求出某1个词对应的最可能的8个上下文词时，我们可以通过一次前向传播算法得到概率大小排前8的softmax概率对应的神经元所对应的词

4. 基于Hierarchical Softmax的模型

  DNN模型的这个处理过程非常耗时，我们的词汇表一般在百万级别以上，这意味着我们的输出层需要进行softmax计算各个词的输出概率的的计算量很大。有没有简化一点点的方法呢？

  __霍夫曼树伪代码__ 

  ```
  输入：权值为(w1,w2,...wn)的n个节点
  输出：对应的霍夫曼树
  
  1）将(w1,w2,...wn)看做是有n棵树的森林，每个树仅有一个节点。
  2）在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。
  3）将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林
  4）重复步骤2）和3）直到森林里只有一棵树为止。
  ```

  - 改进1：对于从输入层到隐藏层的映射，没有采取神经网络的线性变换加激活函数的方法，而是采用简单的对所有输入词向量__求和并取平均__的方法，于是从多个词向量变成了一个词向量
  - 改进2：从隐藏层到输出的softmax层这里的计算量做了改进。为了避免要计算所有词的softmax概率，word2vec采样了__霍夫曼树__来代替从隐藏层到输出softmax层的映射。

  在霍夫曼树中，根节点的词向量对应我们的投影后的词向量，而所有叶子节点就类似于之前神经网络softmax输出层的神经元，叶子节点的个数就是词汇表的大小。隐藏层到输出层的softmax映射沿着霍夫曼树一步步完成的，因此这种softmax取名为`Hierarchical Softmax` 

  如何“沿着霍夫曼树一步步完成”呢？我们采用了二元逻辑回归的方法，即规定沿着左子树走，那么就是负类(霍夫曼树编码1)，沿着右子树走，那么就是正类(霍夫曼树编码0)。判别正类和负类的方法是使用sigmoid函数，即：

  $$
  P(+) = \sigma(x^T_w \theta) = \frac{1}{1+e^{-x^T_w \theta}}\qquad
  $$

5. 基于Negative Sampling模型

   使用霍夫曼树来代替传统的神经网络，可以提高模型训练的效率。但是如果我们的训练样本里的中心词`w`是一个很生僻的词，那么就得在霍夫曼树中辛苦的向下走很久了。能不能不用搞这么复杂的一颗霍夫曼树，将模型变的更加简单呢？

   Negative Sampling摒弃了霍夫曼树，采用了Negative Sampling（负采样）的方法来求解

   比如我们有一个训练样本，中心词是`w`,它周围上下文共有2c个词，记为context(w)。由于这个中心词`w`,的确和context(w)相关存在，因此它是一个真实的正例。通过Negative Sampling采样，我们得到neg个和w不同的中心词 $w_i,i=1,2,..neg$ ，这样context(w)和 $w_i$ 就组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，我们进行二元逻辑回归，得到负采样对应每个词$w_i$的模型参数$\theta_{i}$和每个词的词向量。

#### Skip代码

1. 采用jieba分词，对文本进行分词

2. 将语料中的所有词组成一个列表，构建词频统计，词典及反转词典(把文字转换为数值替代)

   ```python
   def build_dataset(words):
       count = [['UNK',-1]]
       # 统计词频
       count.extend(collections.Counter(words).most_common(vocabulary_size-1))
       # 将单词转为编号(以频数排序的编号)
       dictionary = dict()
       for word,_ in count:
           dictionary[word] = len(dictionary)
       # 遍历单词列表，在dict中转为编号，否则转为编号0(Unkown)
       data = list()
       unk_count = 0
       for word in words:
           if word in dictionary:
               index = dictionary[word]
           else:
               index = 0
               unk_count += 1
           data.append(index)
       count[0][1] = unk_count
   
       # 将dictionary中的数据反转，即可以通过ID找到对应的单词
       reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
       return data,count,dictionary,reverse_dictionary
   ```

3. 构建模型需要的训练数据

   ```python
   def generate_batch(batch_size,num_skips,skip_windows):
       global data_index
       assert batch_size % num_skips == 0
       assert num_skips <= 2*skip_windows
   
       batch = np.ndarray(shape=(batch_size), dtype=np.int32)
       labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
       span = 2*skip_windows+1
       # 建立双向队列的缓冲区，实际上是为了构造bath以及labels
       buffer = collections.deque(maxlen=span)
       for _ in range(span):
           buffer.append(data[data_index])
           data_index = (data_index + 1) % len(data)
   
       #每次循环对一个目标单词生成样本
       for i in range(batch_size // num_skips):
           # buffer中第skip_windows个变量为目标单词
           target = skip_windows
           targets_to_avoid = [skip_windows]
           for j in range(num_skips):
               while target in targets_to_avoid:
                   target = random.randint(0, span - 1)
               targets_to_avoid.append(target)
               batch[i*num_skips+j] = buffer[skip_windows]
               labels[i*num_skips+j,0] = buffer[target]
           buffer.append(data[data_index])
           data_index = (data_index+1)%len(data)
       return batch,labels
   ```

4. 构建模型结构

   ```python
   graph = tf.Graph()
   with graph.as_default():
       train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
       train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
       with tf.device('/device:GPU:1'):
           embeddings = tf.Variable(
               tf.random_uniform([vocabulary_size, embedding_size], 
                                 -1.0, 1.0),name='embeddings')
           embed = tf.nn.embedding_lookup(embeddings,train_inputs)
   
           nce_weights = tf.Variable(
               tf.truncated_normal([vocabulary_size, embedding_size],
                                   stddev=1.0 / math.sqrt(embedding_size)))
           nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
           with tf.name_scope('nce_loss'):
               loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,#权重
                                                   biases = nce_biases,# 偏差
                                                   labels = train_labels,# 标签
                                                   inputs = embed, # 输入向量
                                                   num_sampled = num_sampled,# 负采样个数
                                                   num_classes = vocabulary_size))
               tf.summary.scalar('loss', loss)
           optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
           norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
           normalized_embeddings = embeddings / norm
           init = tf.global_variables_initializer()
           merged = tf.summary.merge_all()
           writer = tf.summary.FileWriter('skip_logs/', tf.get_default_graph())
       saver = tf.train.Saver()
   ```

#### CBOW代码

1. 采用jieba分词，对文本进行分词

2. 将语料中的所有词组成一个列表，构建词频统计，词典及反转词典(把文字转换为数值替代)

3. 构建模型所需数据

   ```python
   def generate_batch(batch_size,skip_windows):
       global data_index
       span = 2 * skip_windows + 1
       batch = np.ndarray(shape=(batch_size,span - 1), dtype=np.int32)
       labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
       # 建立双向队列的缓冲区，实际上是为了构造bath以及labels
       buffer = collections.deque(maxlen=span)
       for _ in range(span):
           buffer.append(data[data_index])
           data_index = (data_index + 1) % len(data)
   
       # 每次循环对一个目标单词生成样本
       for i in range(batch_size):
           # buffer中第skip_windows个变量为目标单词
           target = skip_windows
           targets_to_avoid = [skip_windows]
           col_idx = 0
           for j in range(span):
               if j == span // 2:
                   continue
               batch[i,col_idx] = buffer[j]
               col_idx += 1
           labels[i, 0] = buffer[target]
   
           buffer.append(data[data_index])
           data_index = (data_index + 1) % len(data)
       assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
       return batch, labels
   ```

4. 构建模型结构

   ```python
   graph = tf.Graph()
   with graph.as_default():
       train_inputs = tf.placeholder(tf.int32,shape[batch_size,2*skip_windows])
       train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
       with tf.device('/device:GPU:0'):
           embeddings = tf.Variable(
               tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
           softmax_weights = tf.Variable(
               tf.truncated_normal([vocabulary_size, embedding_size],
                                   stddev=1.0 / math.sqrt(embedding_size)))
           softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
           # 与skipgram不同， cbow的输入是上下文向量的均值，因此需要做相应变换
           embeds = None
           for i in range(2 * skip_windows):
               embedding_i = tf.nn.embedding_lookup(
                   embeddings,train_inputs[:,i])
               emb_x,emb_y = embedding_i.get_shape().as_list()
               if embeds is None:
                   embeds = tf.reshape(embedding_i,[emb_x, emb_y, 1])
                   else:
                       embeds = tf.concat([embeds,tf.reshape(embedding_i
                                                           [emb_x,emb_y,1])],2)
                       assert embeds.get_shape().as_list()[2] == 2 * 
                       skip_windows
                       avg_embed = tf.reduce_mean(embeds,2,keep_dims=False)
                       with tf.name_scope('loss'):
                           loss = tf.reduce_mean(                         
                            tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                                   biases=softmax_biases,
   												labels=train_labels,  
   	                                   			inputs=avg_embed, 
   						                		num_sampled=num_sampled,  
                                           	num_classes=vocabulary_size))
                           tf.summary.scalar('loss', loss)
                           optimizer = tf.train.AdagradOptimizer(0.8).
                           minimize(loss)
                           norm = tf.sqrt(tf.reduce_sum(
                               tf.square(embeddings), 1, keep_dims=True))
                           normalized_embeddings = embeddings / norm
                           init = tf.global_variables_initializer()
                           merged = tf.summary.merge_all()
                           writer = tf.summary.FileWriter(
                               'cbow_logs/', tf.get_default_graph())
                           saver = tf.train.Saver()
   ```

#### TensorFlow 加载 词向量

1. 如果词向量是作为另一个`TensorFlow`模型的一部分进行训练的，使用`tf.train.Saver` 从其他模型检查点文件加载`embedding` 值

```python
# 清空默认图的堆栈
tf.reset_default_graph()
# 初始化 shape = [vocabulary_size,embedding_size]
embedding_var = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],
                                              -1.0,1.0),trainable=False)
# "embeddings"为训练过程中的变量名称
embeddings = tf.train.Saver({"embeddings":embedding_var})
with tf.Session() as sess:
	embeddings.restore(sess,model_path=path)
```

2. 如果使用预训练的词向量，比如文件中存储格式为一词，后加所对应向量，则使用如下方法，主要来自 [训练好的模型](https://spaces.ac.cn/archives/4304/comment-page-1) 和 [使用预训练词向量代码解析](https://blog.csdn.net/lxg0807/article/details/72518962) 

```python
# word2vec词向量装载
def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    line = fr.readline().decode('utf-8').strip()
    #print line
    word_dim = int(line.split(' ')[1])    
    vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print "loaded word2vec"
    fr.close()
    return vocab,embd
vocab,embd = loadWord2Vec(filename)
vocab_size = len(vocab)
embed_dim = len(embd[0])
embedding = np.asarray(embd) # 将结构化数据转为ndarray
# 词向量层
W = tf.Variable(tf.constant(0.0,shape=[vocab_size,embed_dim]),trainable=False,
               name='W')
embedding_placeholder = tf.placeholder(tf.float32,[vocab_size,embed_dim])
embedding_init = W.assign(embedding_placeholder)

# 在网络结构中声明词向量矩阵W
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
```

#### 参考文献

- [博客-词向量Word2vec的本质](https://mp.weixin.qq.com/s/aeoFx6sIX6WNch51XRF5sg)
- [word2vec Parameter Learning Explained论文学习笔记](https://blog.csdn.net/lanyu_01/article/details/80097350) 
- [刘建平 word2vec原理](https://www.cnblogs.com/pinard/p/7160330.html) 
- [中文训练词向量-附源码](https://zhuanlan.zhihu.com/p/28979653) 
- [TensorFlow 使用预训练的词向量](https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow#) 
