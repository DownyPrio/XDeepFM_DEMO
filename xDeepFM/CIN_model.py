import numpy as np
import xDeepFM.LR_model as LR

class CIN_model(object):
    def __init__(self):
        return
    def __init__(self,depth=3,H_per=3):
        self.depth=3
        self.H_per=3
        return
    def divide_col(self,X,X0):
        H=X.shape[0]
        D=X.shape[1]
        m=X0.shape[0]
        X_T=X.T
        X0_T=X0.T
        print(X_T[0].shape)
        print(X0_T)
        print("**********")
        res_map=[]
        for index in range(H):
            X_res=np.zeros((X.shape[0],X0.shape[0]))
            for raw in range(len(X_T)):
                for col in range(len(X0_T)):
                    #print(X_T[raw].reshape(H,1))
                    #print(X0_T[col].reshape(1,m))
                    X_res=np.matmul(X_T[raw].reshape(H,1),X0_T[col].reshape(1,m))
            res_map.append(X_res)
        return res_map
    def filter(self,H,X,X0):
        (x,y)=input.shape
        weights=np.zeros((H,y))
        result=X*weights
        return result
    def predict(self,input,paraset):
        tmp_list=[input]
        flatten_list=[]
        for index in range(self.depth):
            tmp_res=filter(paraset[index].shape[0],tmp_list[-1],input)
            tmp_list.append(tmp_res)
            flatten_list.append(tmp_res)
        flatten_list=np.array(flatten_list).reshape(1,)
        result=LR.perdict(flatten_list)


    def para_init(self,input):
        (x,y)=input.shape
