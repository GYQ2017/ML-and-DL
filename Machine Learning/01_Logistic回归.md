## Logistic回归

- 时间：2018-07-10
- 摘要：首先介绍了Logistic回归的概念和一般过程，然后介绍了三种最优化算法，最后通过实战`从疝气病预测病马的死亡率`练习
- 优缺点：计算代价不高，易于理解和实现；容易欠拟合，分类精度不高

#### 主要思想

假设现在有一些数据点，我们用一条直线对这些点进行拟合（该线称为最佳拟合直线），这个拟合过程就称作__回归__。

Logistic回归进行分类的主要思想是:根据现有数据对分类边界线建立回归公式，以此__进行分类__。训练分类器时的做法就是寻找最佳拟合参数，使用的是最优化算法。

#### Logistic回归一般过程

- 收集数据：采用任意方法收集数据
- 准备数据：需要进行距离运算，因此要求数据类型为数值型
- 分析数据：采用任意方法对数据进行分析
- 训练算法：训练的目的是为了找到最佳的分类回归系数
- 测试算法：一旦训练步骤完成，分类将会很快
- 使用算法：首先，我们需要输入一些数据，并将其转化为对应的结构化数值；接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定他们属于哪个类别

#### 基于最优化方法的最佳回归系数确定

Sigmoid函数的输入为z，由下面公式得出：
$$
z = w_0x_0+w_1x_1+w_2x_2+...+w_nx_n
$$
其中的向量x是分类器的输入数据，向量w也就是我们要找到的最佳参数（系数），从而使分类器尽可能的精确。为了寻找最佳系数，需要用到最优化理论的一些知识。

1. 梯度上升法

梯度上升法基于的思想是：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。梯度上升法到达每个点后都会重新估计移动的方向。从p0开始，计算完该点的梯度，函数就根据梯度移动到下一个点p1。在p1点，梯度再次被重新计算，并沿新梯度方向移动到p2，如此循环迭代，直到满足停止条件。迭代的过程中，梯度算子总是保证我们能选取到最佳的移动方向。梯度上升算法的迭代公式如下：
$$
w := w + \alpha\nabla_wf(w)
$$
其中，梯度下降算法的迭代公式为：$w := w - \alpha\nabla_wf(w)$ ，梯度上升算法用来求函数的最大值，下降算法用于求函数的最小值。

梯度上升法的代码：

```python
每个回归系数初始化为1
重复R次：
	计算整个数据集的梯度
	使用迭代公式更新回归系数的向量
返回回归系数
def gradAscent(dataMatIn, classLabels) :
    dataMatrix = mat(dataMatIn)				# 转换成矩阵
    labelMat = mat(classLabels).transpose() # 将行向量转换为列向量方便矩阵运算
    m, n = np.shape(dataMatrix)
    alpha = 0.001 
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles) :
        # 计算真实类别与预测类别的差值，按照该差值的方向调整回归系数
        h = sigmoid(dataMatrix * weights) 
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
```

2. 随机梯度上升法

梯度上升算法在每次更新回归系数时都需要遍历整个数据集，如果数据太大就不适用。一种改进的方法是一次仅使用一个样本点来更新回归系数，该方法称为__`随机梯度上升算法`__。

```python
所有回归系数初始化为1
对数据集每个样本：
	计算该样本梯度
	使用迭代公式更新回归系数
返回回归系数
def stocGradAscent0(dataMatrix, classLabels) :
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    maxCycles = 500 # 迭代次数
    for i in range(maxCycles) :
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
```

随机梯度上升算法改进版，第一处改进了alpha值，使alpha随着迭代次数不断减少，缓解数据波动；第二处改进了通过随机选取样本来更新回归系数，减少周期性的波动。

```python
def stocGradAscent1(dataMatrix, classLabels, numIter=150) :
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter) :
        dataIndex = list(range(m))
        for i in range(m) :
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])    
    return weights
```

#### 实战-从疝气病预测病马死亡率

具体代码可看项目`Project_ML`

#### 参考文献

- [《机器学习实战》](https://item.jd.com/11242112.html) 