import numpy as np

class LR_model(object):
    def __init__(self):
        pass

    def predict(self,input):
        (x,y)=input.shape
        weights=np.zeros((1,y))+1
        biase=np.zeros((1,1))+1
        pred_result=[]
        tmp_input=input
        pred_result=(np.matmul(tmp_input,weights.T)+biase)
        return pred_result