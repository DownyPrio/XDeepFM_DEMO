import numpy as np
import xDeepFM.LR_model as LR
from xDeepFM import *

class CIN_model(object):
    def __init__(self):
        return
    def __init__(self,depth=3,H_per=3):
        self.depth=3
        self.H_per=3
        print("CIN model initilization completed.")
        return
    def divide_col(self,X,X0):
        H=X.shape[0]
        D=X.shape[1]
        m=X0.shape[0]
        X_T=X.T
        X0_T=X0.T
        # print(X_T[0].shape)
        # print(X0_T)
        # print("**********")
        res_map=[]
        for index in range(D):
            X_res=np.zeros((X.shape[0],X0.shape[0]))
            # for raw in range(H):
            #     for col in range(m):
            #         #print(X_T[raw].reshape(H,1))
            #         #print(X0_T[col].reshape(1,m))
            #         X_res=np.matmul(X_T[raw].reshape(H,1),X0_T[col].reshape(1,m))
            X_res=np.matmul(X_T[index].reshape(H,1),X0_T[index].reshape(1,m))
            res_map.append(X_res)
        return np.array(res_map) #作为inputset输入滤波器，len应为D（测试中为4）
    def filter_demo(self,inputset,filterset):
        filter_matrix=[]
        H_pri=len(inputset)
        # print(inputset.shape)
        inputset=inputset.reshape((H_pri,1,-1))
        # print(inputset.shape)
        H=len(filterset)
        # print(filterset.shape)
        filterset=filterset.reshape((H,-1,1))
        # print(filterset.shape)
        for index in range(len(inputset)):
            # print("//////////")
            # print(index)
            # print("//////////////")
            field_matrix=[]
            for each in filterset:
                # print(inputset[index].shape)
                # print(each.shape)
                field_matrix.append(np.matmul(inputset[index],each))
            # print("field_matrix:")
            # print(field_matrix)
            filter_matrix.append(field_matrix)
        # print(np.array(filter_matrix).shape)
        return np.array(filter_matrix).reshape((H,self.D))

    # def filter(self,H,X,X0):
    #     (x,y)=input.shape
    #     weights=np.zeros((H,y))
    #     result=X*weights
    #     return result
    def predict(self,inputset):
        print("CIN prediction start.")
        # tmp_list=[input]
        # flatten_list=[]
        # for index in range(self.depth):
        #     tmp_res=filter(paraset[index].shape[0],tmp_list[-1],input)
        #     tmp_list.append(tmp_res)
        #     flatten_list.append(tmp_res)
        # flatten_list=np.array(flatten_list).reshape(1,)
        # result=LR.perdict(flatten_list)
        # print("9999999999999")
        # print(inputset.shape)
        self.D=inputset.shape[1]
        flattendCandiList=[inputset]
        flattendFeature=[]
        for index in range(self.depth):
            # print("the depth:")
            # print(index)
            # print(flattendCandiList[-1].shape)
            # print("——————————————————————")
            # print(flattendCandiList[-1].shape)
            (x,y)=flattendCandiList[-1].shape
            x0=len(flattendCandiList[0])

            tmpInputSet=self.divide_col(flattendCandiList[-1],flattendCandiList[0])
            initParaSet=np.zeros((self.H_per,x,x0))+1
            # print("init:")
            # print(initParaSet.shape)
            tmpSet=self.filter_demo(tmpInputSet,initParaSet)
            flattendCandiList.append(tmpSet)
            sumPoolingValue=np.sum(tmpSet,axis=1).T
            # print("tmp:")
            # print(tmpSet)
            flattendFeature.append(sumPoolingValue)
            # print("list len:")
            # print(np.array(tmpSet).shape)
            # print(len(np.array(flattendFeature).reshape(-1,1,1)))
            # print("Depth:"+str(index)+" completed.")
        print("CIN prediction completed.")
        output_res=np.array(flattendFeature).reshape((1,-1))
        return output_res
