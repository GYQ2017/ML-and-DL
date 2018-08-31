### AdaBoost

- 时间：2018-08-31
- 摘要：首先对集成方法进行简单概述，介绍了bagging和boosting；然后着重介绍了boosting的AdaBoost，介绍了运行流程和主要公式；最后实现了基于`sklearn`的AdaBoost的分类
- 优缺点：泛化错误率低、无参数调整；但是对离群点敏感

#### 概述

将不同的分类器组合起来，这种组合方法称作“集成”。使用集成方法时会有很多形式：可以是不同算法的集成，也可以是统一算法不同设置下的集成，还可以是数据集不同部分分配给不同分类器之后的集成。

本文主要介绍基于同一种分类器，多个不同实例的两种计算方法。在这些方法中，数据集会不断变化，而后应用于不同的实例分类器上。

#### 方法1 bagging

bagging是一种基于数据随机重抽样的分类器构建方法，是从原始数据集选择S次后得到S个新数据集的一种技术。新数据集和原始数据集大小相等。每个数据集是通过在原始数据集中随机选择一个样本进行替换得到的，即新数据集可有重复的值，也可有一些值未在新集合中出现。

在S个数据集建好之后，将学习算法分别作用于不同的数据集就得到了S个分类器。当进行分类时，选择分类器__投票结果__中最多的类别作为最后的分类结果

#### 方法2 boosting

boosting与方法1类似，但是boosting是通过关注被已有分类器错分的那些数据来获得新的分类器，即每个分类器都根据已训练出的分类器的性能来进行训练。除此之外，boosting分类的结果是基于所有分类器的加权求和，boosting中分类器权重并不相等。本文着重介绍boosting的最流行的版本-__AdaBoosting__ 

- AdaBoosting 训练算法:基于错误提升分类器的性能

  运行过程如下：

  - 训练数据集中的每个样本，并赋予其一个权重，这些权重构成了向量D。一开始，这些权重都初始化成相等值
  - 在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据上再次训练弱分类器。在分类器的第二次训练中，会重新调整每个样本的权重，分对的样本的权重会降低，分错的样本的权重会提高
  - AdaBoosting为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器的错误率进行计算的

  ![AdaBoost1](/home/gaoyinquan/MS/Machine Learning/Picture_ML/AdaBoost1.png)

                     图解：图中左边是数据集，直方图的不同宽度表示每个样例上不同权重

  alpha计算公式：$\alpha=\frac{1}{2}ln(\frac{1-\varepsilon}{ \varepsilon})$  

  计算出alpha值之后，对权重向量D进行更新：

  若样本被正确分类，那么权重更改为：$D_i^{t+1}=\frac{D_i^t e^{- \alpha }}{Sum(D)}$   若样本被错误分类，那么权重更改为：$D_i^{t+1}=\frac{D_i^t e^{ \alpha }}{Sum(D)}$ 

  在计算出D之后，AdaBoost又开始进入下一轮迭代，不断重复训练和调整权重的过程，直到训练错误率为0或弱分类器的数目达到用户的指定值为止

- 算法流程

  - 输入：训练数据集$T={(x_1,y_1),...,(x_N,y_N)}$ ，迭代次数M

  1. 初始化训练样本的权值分布：$D_1=(w_{1,1},...,w_{1,i})$ 

  2. 对于 $m=1,2,....M$ :

     a. 使用具有权值分布$D_m$ 的训练数据集进行学习，得到弱分类器$G_m(x)$ 

     b. 计算 $G_m(x)$ 在训练数据集上的分类误差率

     c. 计算 $G_m(x)$ 在分类器中所占权重$\alpha$ 

     d. 更新训练数据集的权值分布

  3. 得到最终分类器

#### Sklearn AdaBoost

基于`sklearn` 的 AdaBoost 包含回归和分类两个模型，本文实现分类“从疝气病症预测病马的死亡”，基本知识可参考[刘建平Pinard](https://www.cnblogs.com/pinard/p/6136914.html) 

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def loadData(filename):
    dataMat = [];      labelMat = []
    with open(filename) as fr:
        numFeat = len(fr.readline().split('\t'))
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

if __name__ == '__main__':
    dataMat,labelMat = loadData('../01_Logistic/horseColicTraining.txt')
    print(len(dataMat))

    Ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),
                             n_estimators=400,learning_rate=0.1)
    Ada.fit(dataMat,labelMat)

    TestDataMat,TestlabelMat = loadData('../01_Logistic/horseColicTest.txt')
    acc_test = Ada.score(TestDataMat,TestlabelMat)
    acc_train = Ada.score(dataMat,labelMat)
    print('acc_train:%.5f,acc_test:%.5f'%(acc_train,acc_test))
```

#### 参考文献

- [机器学习实战](https://www.amazon.cn/gp/search?index=books&keywords=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98&tag=readfreeme-23) 
- [刘建平Pinard 实战](https://www.cnblogs.com/pinard/p/6136914.html) 
- [刘建平Pinard 原理](https://www.cnblogs.com/pinard/p/6133937.html) 


   
