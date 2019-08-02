import xDeepFM.embedding_layer as emd
import numpy as np

inputset=np.array([[0,0,0,0,1],[0,1,0],[0,1,0,0,1,0,0,0]])
outemd=emd.embedding_layer(4).embedding(inputset)
print(outemd)