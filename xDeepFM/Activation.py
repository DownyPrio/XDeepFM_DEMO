from xDeepFM import *

class Activation(object):
    def __init__(self,activation_function="sigmoid"):
        self.activation_function=activation_function
        return
    def func(self,input):
        if self.activation_function=="sigmoid":
            return self.sigmoid(input)
        elif self.activation_function=="relu":
            return self.relu(input)
        else:
            print("Please select activaion functions!")

    def sigmoid(self,input):
        return 1/(1+np.exp(-input))
    def relu(self,input):
        return (np.abs(input)+input)/2
    def tanh(self,input):
        return np.tanh(input)
