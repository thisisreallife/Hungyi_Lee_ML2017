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















