import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 进行矩阵乘法和加法运算
def forward(x, w1, w2, b1, b2,):
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    y = sigmoid(a2)
    return y