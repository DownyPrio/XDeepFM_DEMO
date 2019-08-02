import numpy as np

class DNN_model(object):
    def __init__(self):
        return
    def __init__(self,layer_num=2,nodes_per=2):
        self.layer_num=layer_num
        self.nodes_per=nodes_per
        self.weights=np.zeros((layer_num,nodes_per,nodes_per))+1
        self.biase=np.zeros((layer_num,1,nodes_per))+1
        print("DNN model initilization completed.")
        # print(self.weights)
        # print(self.biase)

    def discribe(self):
        print("layers_number is:"+str(self.layer_num))
        print("nodes_num is:"+str(self.nodes_per))+1

    def fit(self,trainset,labelset):
        (x,y)=trainset.shape
        Weights=np.array(y*[1])
        print(Weights)
        Bias=np.array(3)
        tmp_result=np.dot(trainset,Weights)+Bias
        return tmp_result
    def __lr(self,input,para):
        (x,y)=input.shape
        pred_result=[]
        tmp_input=input
        for each in range(len(para)):
            # print("weight：")
            # print(str(para[0][each]))
            # print("biase:")
            # print(para[1][each])
            pred_result.append(np.dot(tmp_input,para[0][each])+para[1][each])
            #print("tmp result:"+str(pred_result))
        return pred_result

    def __input_layer(self,testset):
        # print(testset.shape)
        para_weights=np.zeros((testset.shape[1],self.nodes_per))+1
        para_biase=np.zeros((1,self.nodes_per))+1
        # print(testset)
        # print(para_weights.T)
        return np.matmul(testset,para_weights)+para_biase
    def __hiden_layer(self,testset):
        tmp_result=testset
        #print(testset)
        #print("*****************")
        for each in range(len(self.weights)-1):
            tmp_result=np.matmul(tmp_result,self.weights[each].T)+self.biase[each]
            #print(tmp_result)
            #print("***************")
        return tmp_result
    def __out_layer(self,inputset):
        out_weights=np.zeros((1,self.nodes_per))+1
        out_biase=np.zeros((1,1))+1
        return np.matmul(inputset,out_weights.T)+out_biase
    def predict(self,input):
        print("DNN prediction start.")
        return self.__hiden_layer(self.__input_layer(input))
        print("DNN prediction completed.")
        #return self.__out_layer(self.__hiden_layer(self.__input_layer(input)))



