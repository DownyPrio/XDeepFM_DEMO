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
        return np.array(res_map)
    def filter_demo(self,inputset,filterset):
        filter_matrix=[]
        H_pri=len(inputset)
        print(inputset.shape)
        inputset=inputset.reshape((H_pri,1,-1))
        print(inputset.shape)
        H=len(filterset)
        print(filterset.shape)
        filterset=filterset.reshape((H,-1,1))
        print(filterset.shape)
        for index in range(len(inputset)):
            field_matrix=[]
            for each in filterset:
                print(inputset[index])
                print(each)
                field_matrix.append(np.matmul(inputset[index],each))
            filter_matrix.append(field_matrix)
        return filter_matrix

    # def filter(self,H,X,X0):
    #     (x,y)=input.shape
    #     weights=np.zeros((H,y))
    #     result=X*weights
    #     return result
    def predict(self,inputset):
        # tmp_list=[input]
        # flatten_list=[]
        # for index in range(self.depth):
        #     tmp_res=filter(paraset[index].shape[0],tmp_list[-1],input)
        #     tmp_list.append(tmp_res)
        #     flatten_list.append(tmp_res)
        # flatten_list=np.array(flatten_list).reshape(1,)
        # result=LR.perdict(flatten_list)
        flattendCandiList=[inputset]
        flattendFeature=[]
        for index in range(self.depth):
            print(flattendCandiList[-1].shape)
            (x,y)=flattendCandiList[-1].shape
            tmpInputSet=self.divide_col(flattendCandiList[-1],flattendCandiList[0])
            initParaSet=np.zeros((self.H_per,x,y))+1
            print("init:")
            print(initParaSet.shape)
            tmpSet=self.filter_demo(tmpInputSet,initParaSet)
            sumPoolingValue=np.sum(tmpSet,axis=0)
            print("tmp:")
            print(tmpSet)
            flattendFeature.append(sumPoolingValue)
            print("Depth:"+str(index)+" completed.")
        return np.array(flattendFeature).reshape((-1,1,1))





    def para_init(self,input):
        (x,y)=input.shape
