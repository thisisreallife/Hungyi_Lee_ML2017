import numpy as np
import random

class LinearRegressionGD(object):
    # 初始化变量
    def __init__(self,eta = 0.001, n_iter = 20):
        '''
        :param eta:  learning rate
        :param n_iter: n_iter表示训练的次数，每一次对一批的所有样本进行同时训练
        '''
        self.eta = eta
        self.n_iter = n_iter

        def net_input(self, X):
            '''
            :param self:
            :param X:
            :return: y=w_0x_0+w_1x_1+...+w_nx_n
            '''
            # 数据类型是Numpy数组
            # np.dot(a,b) 如果a为一维的向量，b为一维的向量，得到的是向量点积
            # np.dot(a,b) 如果a为二维矩阵，b为一维向量，这时b会被当做一维矩阵来计算
            # np.dot(a,b) 如果a和b都是二维矩阵，此时dot就是进行的矩阵乘法运算
            return np.dot(X, self.w_[1:]) + self.w_[0]


        def batch(self, X,y):
            '''
            :param self:
            :param X: 样本数据特征X
            :param y: 每个样本对应的真实标签值y
            :return: cost
            '''
            self.w = np.zeros(1 + X.shape[1])   # 得到的是一维数组
            # self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
            self.cost = []  # 用来保存每一次的预测误差，cost应该是越小越好

            for i in range(self.n_iter):
                output = self.net_input()  # 预测的结果
                errors = (y - output)
                self.w[1:] += self.eta * np.dot(X.T, errors)   # 第一个矩阵的列数和第二个矩阵的行货数相同才有意义
                self.w[0] += self.eta * errors.sum()  # 直接sum()函数求和
                cost = (errors ** 2).sum() / 2.0
                self.cost.append(cost)
            return self.cost

        def SGD(self,X,y):
            self.w = np.zeros(1 + X.shape[1])
            self.cost = []
            m = X.shape[0]
            for i in range(m):
                output = self.net_input
                errors = (y - output)
                self.w[1:] += self.eta * errors * X[i]
                self.w[0] += self.eta * errors.sum()
                cost = (errors ** 2).sum() /2.0
                self.cost.append(cost)
            return self.cost

        def miniBatch(self, X,y, n_iter = 200):
            self.w = np.zeros(1 + X.shape[1])
            self.cost = []

            m,n = np.shape(X)
            dataIndex = len(m)  # 获取数据集行下标列表
            for j in range(n_iter):  # 迭代次数
                for i in range(m):   # 遍历行列表
                    # alpha在每次迭代的时候都会调整，这个缓和数据波动或者高频波动
                    # 虽然alpha会随着迭代次数不断减小，但永远不会减小到0，这是因为此项还存在一个常数项
                    # 这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响
                    alpha = 4 / (1.0 + j + i) + 0.01
                    # 通过随机选取样本来更新回归系数，这种方法将减少周期性的波动
                    randIndex = int(random.uniform(0, len(m)))  # 随机获取样本
                    output = self.net_input
                    errors = (y-output)
                    self.w[1:] += self.eta * errors * X[randIndex]
                    self.w[0] += self.eta * errors.sum()
                    # 这种方法随机从列表中选出一个值，然后从列表中删掉该值
                    del(X[randIndex])
                    cost = (errors ** 2).sum() / 2.0  # array.sum()对array的全部元素进行相加
            return self.cost



# 划分数据
def train_test_split(x,y):
    split_index = int(len(y) * 0.7)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# 归一化
# 归一化是建立在划分完数据之后的
x_train = (x_train - np.min(x_train, axis=0))/ (np.max(x_train,axis=0)-np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0))/ (np.max(x_test,axis=0)-np.min(x_test, axis=0))














