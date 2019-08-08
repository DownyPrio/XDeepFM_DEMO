import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidDerivationx(y):
    return y * (1 - y)


if __name__ == '__main__':
    # 初始化一些参数
    alpha = 0.5
    numIter = 1000000 #迭代次数
    w1 = [[0.15, 0.20], [0.25, 0.30]]  # Weight of input layer
    w2 = [[0.40, 0.45], [0.50, 0.55]]
    # print(np.array(w2).T)
    b1 = 0.35
    b2 = 0.60
    x = [0.05, 0.10]
    y = [0.01, 0.99]
    # 前向传播
    z1 = np.dot(w1, x) + b1     # dot函数是常规的矩阵相乘
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    for n in range(numIter):
        # 反向传播 使用代价函数为C=1 / (2n) * sum[(y-a2)^2]
        # 分为两次
        # 一次是最后一层对前面一层的错误

        delta2 = np.multiply(-(y-a2), np.multiply(a2, 1-a2))
        # for i in range(len(w2)):
        #     print(w2[i] - alpha * delta2[i] * a1)
        #计算非最后一层的错误
        # print(delta2)
        delta1 = np.multiply(np.dot(np.array(w2).T, delta2), np.multiply(a1, 1-a1))
        # print(delta1)
        # for i in range(len(w1)):
        # print(w1[i] - alpha * delta1[i] * np.array(x))
        #更新权重
        for i in range(len(w2)):
            w2[i] = w2[i] - alpha * delta2[i] * a1
        for i in range(len(w1)):
            w1[i] = w1[i] - alpha * delta1[i] * np.array(x)
        #继续前向传播，算出误差值
        z1 = np.dot(w1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        print(str(n) + " result:" + str(a2[0]) + ", result:" +str(a2[1]))
        # print(str(n) + "  error1:" + str(y[0] - a2[0]) + ", error2:" +str(y[1] - a2[1]))