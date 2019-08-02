import numpy as np
import xDeepFM.LR_model as LR

class embedding_layer(object):
    def __init__(self):
        return
    def __init__(self,fieldD):
        self.fieldD=fieldD
        print("embedding layer initilization completed.\nthe embedding dimension is {}.".format(self.fieldD))
    def embedding(self,inputSparse):
        print("embedding start.")
        outEmbedding=[]
        for each in inputSparse:
            W=np.zeros((len(each),self.fieldD))+1
            B=np.zeros((1,self.fieldD))+1
            outEmbedding.append(np.matmul(each,W)+B)
        #print(np.array(outEmbedding).shape)
        print("embedding end.")
        return np.array(outEmbedding)
