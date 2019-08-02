import numpy as np

class LR_model(object):
    def __init__(self):
        pass

    def predict(self,input):
        print(input)
        y=len(input)
        weights=np.zeros((1,y))+1
        biase=np.zeros((1,1))+1
        pred_result=[]
        tmp_input=np.array([input])
        print(tmp_input)
        print(weights.T)
        pred_result=(np.matmul(tmp_input,weights.T)+biase)
        return np.array(pred_result)