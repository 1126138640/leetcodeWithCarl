import numpy as np

# 定义神经网络的参数
W = np.random.rand(3, 2)  # 权重矩阵
X = np.random.rand(2, 1)  # 输入矩阵
Y = np.random.rand(3, 1)  # 输出矩阵

# 前向传播
Z = np.dot(W, X)  # 线性变换
A = np.tanh(Z)  # 激活函数

# 计算梯度
dZ = np.multiply(1 - np.power(A, 2), Y)  # 损失函数对Z的梯度
dW = np.dot(dZ, X.T)  # 损失函数对W的梯度

# 打印梯度
print("dW =", dW)

