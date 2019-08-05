from xDeepFM import *
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
        return np.array(list[map(relu,input)])
    def sigmoid_derivative(self,input):
        return input*(1-input)
    def backPropagation(self,trainData,model,learing_rate=0.1,epochs=10):
        model=DNN.DNN_model()
        W_para=model.weights
        B_para=model.biase
        for epoch in range(epochs):
            print("epochs:{}/{}".format(str(epoch),str(epochs)))
            for each in trainData:
                trainY=model.predict(each[:-1])#差一个输出层
                loss=each[-1]*np.log(trainY)+(1-each[-1])*np.log(1-trainY)
                '''
                输出层更新
                '''
                sigma=loss*self.sigmoid_derivative(model.resLayer[-1])
                W_para[-1]-=learing_rate*sigma*model.resLayer[-2]
                B_para[-1]-=learing_rate*sigma
                '''
                隐藏层更新
                '''
                for n in range(model.layer_num-2,-1,-1):
                    sigma[-n]=W_para[-(n-1)]*sigma[-(n-1)]*self.relu_derivative(model.resLayer[-n])
                    W_para[-n]-=learing_rate*sigma*model.resLayer[-(n+1)]
                    B_para[-n]-=learing_rate*sigma
            model.Weights=W_para
            model.Biase=B_para



