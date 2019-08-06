import numpy as np

class LR_model(object):
    def __init__(self):
        print("LR model initilization completed")
        pass

    def predict(self,input):
        print("LR prediction start")
        #input是shape为（1,n）的稀疏向量
        #初始化shape为（n,1）的权重向量和（1,1）的biase
        self.weights=np.zeros((input.shape[1],1))+1
        self.biase=np.zeros((1,1))+1
        #result.shape=（1,1）
        result=np.matmul(input,self.weights)+self.biase
        return result[0][0]