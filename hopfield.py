import numpy as np
import matplotlib.pyplot as plt

class Hopfield:

    def __init__(self, pre_result):
        self.dim = pre_result['neuron_num']
        self.graph_width = pre_result['graph_width']
        self.graph_length = self.dim // self.graph_width

    def calculate_W(self, x_inputs):
        """ X = [[1, -1, 1], [-1, 1, 1], ...]"""
        self.W = np.zeros((self.dim, self.dim))
        for line in x_inputs:
            x = np.matrix(line)
            self.W += np.matmul(x.transpose(), x)
        self.W = (self.W - len(x_inputs) * np.identity(self.dim)) / self.dim

    def calculate_threshold(self):
        temp = np.ones((self.dim, 1))
        self.threshold = np.matmul(self.W, temp)
    
    def sign(self, x, u):
        """ x is np.matrix """
        # v = u - self.threshold
        v = u

        for i in range(self.dim):
            if v[i, 0] > 0:
                x[0, i] = 1
            elif v[i, 0] < 0:
                x[0, i] = -1
        return x

    def graph(self, x):
        """ x is np.matrix """
        for i in range(self.graph_width):
            for j in range(self.graph_length - 1, -1, -1):
                if x[0, self.graph_width * (self.graph_length - 1 - j) + i] == 1:
                    g = 'ks'
                    plt.plot(i, j, g, markersize = 10)
                else: 
                    continue
    
    def update_x(self, x):
        """ x is 1-D array """
        x = np.matrix(x)
        u = np.matmul(self.W, x.transpose())
        new_x = self.sign(x, u)
        return new_x

    def calculate_accuracy(self, x, new_x):
        Comparison = np.equal(x, new_x)
        self.accuracy = round(np.sum(Comparison) / self.dim, 3)

    def equalty(self, x, y):
        return np.array_equal(x, y)



 