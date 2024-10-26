import numpy as np
def sigmoid(x):
 return 1/(1 + np.exp(-x))

class Network():
    def __init__(self, k, n, init_weights=None, init_biases=None):
        min_val = 0
        max_val = 1
        # weights.shape (k, n)
        # n is the amount of input neurons

        self.weights = init_weights if init_weights is not None else np.random.uniform(min_val,max_val,(k,n))
        self.biases = init_biases if init_biases is not None else np.random.uniform(min_val,max_val,(k,1))
        print(self.weights)
        print(self.biases)
        assert(len(self.weights) == len(self.biases))

    def feedforward(self, a):
        return np.array([sigmoid(np.dot(w,a) + b) for w,b in zip(self.weights, self.biases)])

    def feedforward_v2(self, a):
        if len(a.shape) == 1:
            a = a.reshape(-1,1)
        assert(self.weights.shape[1] == a.shape[0])
        z = self.weights@a + self.biases
        return sigmoid(z)    

# red de 1 capa,
# k=2 neuronas
# n=3 datos de input

nn = Network(k=2, n=3)
x = np.array([4,5,6])

print(nn.feedforward(x))
print(nn.feedforward_v2(x))