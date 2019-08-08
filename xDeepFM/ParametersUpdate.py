from xDeepFM import *
import numpy as np
'''
参数优化类
'''
class ParametersUpdate(object):
    def __init__(self):
        return
    def relu_derivative(self,input):
        def relu(tmp):
            if tmp==0:
                return 0
            else:
                return 1
        return np.array(list(map(relu,input[0])))
    def sigmoid_derivative(self,input):
        return input*(1-input)
    def backPropagation(self,trainData,model,learing_rate=0.1,epochs=10):
        model=DNN.DNN_model()
        for epoch in range(epochs):
            print("epochs:{}/{}".format(str(epoch),str(epochs)))
            sigma=[]
            for each in trainData:
                label=each[-1]
                each=np.array(each[:-1]).reshape((1,-1))

                # print(each)
                trainY=model.predict(each,0)
                W_para=model.weights
                B_para=model.biase
                print("__________")
                print(W_para)
                print("--------------")
                # print("res:")
                # print(len(model.resLayer))
                # print("/////////////")
                # print(trainY)#差一个输出层
                # print(label)
                loss=label*np.log(trainY)+(1-label)*np.log(1-trainY)
                '''
                输出层更新
                '''
                # print("loss:")
                # print(loss)
                # print("der:")
                # print(self.sigmoid_derivative(model.resLayer[-1]))
                sigma.append(loss*self.sigmoid_derivative(model.resLayer[-1])[0])
                # print(W_para[-1])
                # print(sigma)
                # print(model.resLayer[-1])
                # print(learing_rate*sigma*model.resLayer[-2])
                W_para[-1]-=learing_rate*sigma[-1]*model.resLayer[-2]
                print(W_para[-1].shape)
                B_para[-1]-=learing_rate*sigma[-1]
                '''
                隐藏层更新
                '''
                for n in range(model.layer_num-2,-1,-1):
                    print(-(n-1))
                    print(W_para[-1].T)
                    a=W_para[-1].T
                    print(sigma[-1])
                    b=sigma[-1]
                    c=np.matmul(a,b)
                    #print(np.matmul(a,b))
                    sigma.append(c*self.relu_derivative(model.resLayer[-n])[0])
                    W_para[-n]-=learing_rate*sigma[-1]*model.resLayer[-(n+1)]
                    print(B_para[-n])
                    print(sigma[-1])
                    B_para[-n]-=learing_rate*sigma[-1].reshape((1,-1))
            model.weights=W_para
            model.biase=B_para



