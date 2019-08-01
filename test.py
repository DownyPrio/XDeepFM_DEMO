import xDeepFM.DNN_model as DNN
import numpy as np

print(np.array([[0. ,0.],
                [0. ,0.]]).shape)
one=np.zeros((1,2))+1
print(np.matmul([[1,2]],np.array([[0. ,0.],
                                  [0. ,0.]]))+one)
