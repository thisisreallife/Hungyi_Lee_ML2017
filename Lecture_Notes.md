## Lecture 0 Introduction to ML

人工设定规则的模型，效果一定不会超过设定它的人类，不能进行自我学习。

机器学习是，从数据中归纳出决策的函数。比如在语音识别领域，函数的具体形式太过于复杂，以至于人类无法很好地抽象出来，需要机器自己学习。

机器学习框架：

1. 首先给出一个函数的集合，包含成千上万的函数，称为model（function set）
2. 给出判断准则，定义一个损失函数
3. 用一个有效的运算方法，找到给定规则下最好的function
4. 使用新的数据去测试function表现

机器学习分类：

1. 按照label：
   1. 有监督
   2. 半监督，例如：transfer Learning
   3. 无监督
2. 按照输出不同：
   1. 回归问题
   2. 分类问题
   3. structure learning

关于有监督学习和强化学习的对比，有监督学习learn from teacher，强化学习learn from critics。比如alpha go = supervise learning（人类棋谱） + reinforcement learning（机器对战）。

## Lecture 1 Regression

举了神奇宝贝的例子，研究净化前后能力值的关系。

- step1：Model

- step2：Goodness of function

  Loss function：

  - input： model(function set)

  - output: how bad it is

  Select a best function

- step3：Gradient Descent 

  要求Loss对于参数可微分

  学习率learning rate

  挑选初始化参数值

  从初始化参数值开始计算，达到停止条件后停止。

  可能需要考虑到的问题：

  - 学习率的大小
  - 特征需要标准化，让算法收敛更快
  - 是否是凸优化，有无局部最小值

模型复杂度：

提升模型复杂度会带来训练集上的效果提升，但是未必会有测试集上的提升。需要用验证集去确定。

之前收集到的数据很少，只有十个数据，当收集更多神奇宝贝的数据后，可视化发现，不同的物种，能力值进化曲线不一样。需要哑变量回归。

惩罚：

$\lambda$是惩罚的参数：

- $\lambda$越大模型越平滑，考虑训练误差越少。
- 我们希望平滑一些，但是不要太平滑，所以需要调整$\lambda$
- 不需要对截距项进行惩罚，因为惩罚截距项不影响函数的平滑度

## Lecture 1 demo

有很多进行优化计算的方式：

<http://ruder.io/optimizing-gradient-descent/index.html>

## Lecture 2 Where does the error come from

我们不知道真是的模型函数$f$，对于$f$的估计记为$\hat{f}$，误差的来源是方差和偏差。我们需要找到一个方差和偏差都比较好的估计，这和统计学中估计量中最小方差无偏估计的思想类似。

如果模型复杂度上升，那么模型的方差会变大，偏差变小。

![1557750520826](.\image\Lec02 - 01.png)

关于这个图的理解，见<https://www.jianshu.com/p/8c7f033be58a>，<https://www.cnblogs.com/zongfa/p/9502470.html>。

model = function set，不是最终拟合好的，用来预测的function。

惩罚项的加入能够降低方差，偏差的影响不确定，要看是否包含了真实模型。

![1557754050841](.\image\Lec02 - 02.png)

使用training set和validation set去建模，testing set只用来测试，不能调节参数。

![1557757038673](.\image\Lec02 - traintest.png)

![1557757072278](.\image\Lec02 - CV)



## Lecture 3 Gradient descent 

$$\theta^i = \theta^{i-1} - \eta\nabla \mathrm L(\theta^{i-1})$$

![1557759763492](.\image\lec03 - learning_rate.png)

如何调节学习速率？

![1557761011014](.\image\lec03 - second_derivative)

<https://www.cnblogs.com/pinard/p/5970503.html>







