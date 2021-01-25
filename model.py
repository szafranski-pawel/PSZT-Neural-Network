#!/usr/bin/env python3
import random
import numpy as np

##############################
# Loggers
##############################
class Term:
    red =   '\x1b[31m'
    blue = '\x1b[34m'
    green = '\x1b[32m'
    default = '\x1b[0m'

def error(*args):
    args = ('[' + Term.red + '!!!' + Term.default + ']',) + args
    print(*args)

def info(*args):
    args = ('[' + Term.green + '+' + Term.default + ']',) + args
    print(*args)


##############################
# Neural layers
##############################
class FullLayer:
    # Implementation of layer in which all nodes are connected
    def __init__(self, inputCnt, outputCnt, learnRate):
        self.weights = np.matrix([
            [random.uniform(-1, 1) for _ in range(inputCnt)] for _ in range(outputCnt)
        ])
        
        self.biases = np.array([random.uniform(-1, 1) for _ in range(outputCnt)])
        self.learnRate = learnRate

    def feed(input):
        self.old_input = input
        return np.dot(input, self.weights) + self.biases

    def back(self, output):
        grad = np.dot(output, self.weights.T)

        grad_weights = np.dot(self.old_input.T, output)
        self.weights = self.weights - self.learn_rate * grad_weights

        grad_biases = output.mean(axis=0) * self.old_input.shape[0]
        self.biases = self.biases - self.learn_rate * grad_biases

        return grad

class ActivationFunc:
    def __init__(self):
        pass
    
    def feed(self, input):
        self.old_input = input
        # TODO: Activate

        return input
    
    def back(self, output):
        # TODO: Activate
        
        raise output

##############################
# Nerual model
##############################
class Model:
    def __init__(self, inputCnt, outputCnt, hiddenCnts, learnRate):
        layersSizes = [inputCnt] + hiddenCnts + [outputCnt] 
        self.layers = []

        # Add all layers except the last one
        for i in range(len(layersSizes) - 1):
            self.layers.append(FullLayer(layersSizes[i], layersSizes[i+1], learnRate))
            if i == len(layersSizes) - 1:
                # TODO: Other activator for last layer
                self.layers.append(ActivationFunc())
            else:
                self.layers.append(ActivationFunc())

    def train():
        pass

    def predict(self, input):
        for layer in self.layers:
            input = layer.fead(input)
        return input

    def verify(self, input, correctOutput):
        # TODO: Input and correctOutput are matrices
        predicted = self.predict(input)
        return -correctOutput * predicted
    
    def score(self, input, correctOutput):
        # TODO: Input and correctOutput are matrices
        return self.predict(input) == correctOutput
    

##############################
# Data load and entry point
##############################
def main():
    model = Model(26, 1, [5, 5], 0.1)

if __name__ == "__main__":
    main()
