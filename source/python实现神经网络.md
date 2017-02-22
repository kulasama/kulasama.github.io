[原文地址](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)

本文中，我们将从零开始实现一个简单的三层神经网络, 我会直观的解释怎么做，不需要你掌握太多数学知识。我会指出相关的知识细节。

这里假定你熟悉基本的微积分和机器学习的概念，例如你知道分类器和正则化， 最好能知道类似梯度下降法一类的优化技巧。不过就算是你完全不懂也没关系 :)

假设你计划未来使用类似pyBrain一类的神经网络库，那我们为什么还要从零开始实现一个神经网络呢？  从零开始实现一个神经网络始终是一个有价值的练习， 他能帮助你更深入的理解神经网络是怎么工作的。 这对于设计更有效的神经网络模型非常重要。

需要注意的是这些代码例子执行效率并不高，他们只是很容易理解。在下一篇文章里我将使用theano来实现一个更有效的神经网络。

###数据集生成


我们先开始生成一些我们可以测试的数据集， 幸好我们有scikit-learn这样的代码库可以很方便生成测试数据集。 这样我们不用自己写代码来生成，而是简单的调用make_moons函数。


> pip install scipy
> pip install numpy
> pip install scikit-learn
> pip install matplotlib


```py
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)
x, y = datasets.make_moons(200, noise=0.20)
plt.scatter(x[:,0], x[:,1], s=40, c=y,cmap=plt.cm.Spectral)
plt.show()
```
我们生成了两类数据， 红色的点和蓝色的点。你可以把红色的点想象为男病人， 蓝色的点想象为女病人.  x轴和y轴是生理参数.

我们的目标是实现一个可以通过给定生理指出正确性别的分类器。注意这些数据不是线性可分的。我们不能去画一条直线来区分性别。这意味着这些数据不能使用线性回归模型（例如逻辑回归）。 除非你手工寻找数据的非线性特征。
事实上这就是神经网络的主要优点之一， 你不需要操心特征工程。隐藏的神经网络层将为你自动挖掘特征。

###逻辑回归

为了证明前面的观点，我们先训练一个逻辑回归分类器。输入x,y 输出预测的分类（0或者1）。为了简单 我们直接使用scikit-learn里的逻辑分类类


```py
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
```

```py
from sklearn import linear_model
clf = linear_model.LogisticRegressionCV()
clf.fit(x, y)
plot_decision_boundary(lambda x: clf.predict(x), x,y)
plt.title("Logistic Regression")
plt.show()
```

这张图显示了逻辑回归的决策边界。 逻辑回归可以尽可能好的用直线分割数据，但它不能捕捉数据的月型区间。

现在让我们用一个输入层构建一个三层神经网络, 一个输入层， 一个隐藏层还有一个输出层。我们的数据纬度决定了输入层的节点数量,2 同样的，我们的分类总数决定了我们输出层的节点数量。同样是2.（我们尽量避免使用一个输出节点输出0或者1。而使用两个输出节点。使得这个神经网络会更容易扩展到更多场景）。 输入网络接受x坐标和y坐标， 输出两个概率。为女性的概率和为男性的概率。 

我们可以为隐藏层选择维度（节点数），隐藏层里放置更多的节点 我们能拟合更复杂的函数。但是维度越高，代价越大。 首先，需要更多的计算去预测和学习网络参数。 参数越多意味着我们越容易过拟合我们的数据。

如何选择隐藏层的大小呢？ 虽然有一些普遍性的建议和方针，但它总是依赖你的具体问题。比起科学来这更像是一门艺术。我们将通过修改隐藏层节点数量来看它如何影响我们的输出结果。

我们同样需要为我们的隐藏层准备一个激活函数。激活函数将输入转化为输出。一个非线性激活函数允许我们拟合非线性假设. 常见的激活函数有tanh, sigmoid, 或者ReLUs. 我们将使用在很多场景下表现不错的tanh。这些函数有一个很好的特性， 就是他们的导数可以用原始的函数值计算出来。 例如 $tanh x$的导数是  $1-\tanh^2 x$。这很有用，因为我们可以只用计算$tanh x$一次。之后求导时就可以复用这个值

