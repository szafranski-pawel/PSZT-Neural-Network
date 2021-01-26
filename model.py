#!/usr/bin/env python3
import argparse
import random
import numpy as np
import pandas
import argparse
import sys


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
    # Implementation of layer in which all nodes from previous layer are connected to every node in next layer
    def __init__(self, inputCnt, outputCnt, learnRate):
        self.weights = np.matrix([
            [random.uniform(-1, 1) for _ in range(outputCnt)] for _ in range(inputCnt)
        ])
        # create vector of random biases
        self.biases = np.array([random.uniform(-1, 1) for _ in range(outputCnt)])
        self.learnRate = learnRate

    def feed(self, input):
        self.old_input = input
        return np.dot(input, self.weights) + self.biases

    def back(self, output):
        grad = np.dot(output, self.weights.T)

        grad_weights = np.dot(self.old_input.T, output)
        self.weights = self.weights - self.learnRate * grad_weights

        grad_biases = output.mean(axis=0) * self.old_input.shape[0]
        self.biases = self.biases - self.learnRate * grad_biases

        return grad

# reLU
class HiddenActivationFunc:    
    def feed(self, input):
        self.old_input = input
        return np.maximum(0, input)
    
    def back(self, output):     
        return output * (self.old_input.T > 0)

# Sigmoid
class OutputActivationFunc:
    def feed(self, input):
        self.old_input = input
        return 1/(1+np.exp(-input))
    
    def back(self, output):      
        return (1/(1+np.exp(-output))) * (1 - 1/(1+np.exp(-output)))

##############################
# Score functions
##############################
def lossFunction(predicted, correct):
    predictions = np.clip(predicted, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
    batch_size = predictions.shape[0]
    ce = -np.sum(correct * np.log(predictions)) / batch_size
    return ce

def lossGradient(predicted, correct):
    predictions = np.clip(predicted, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
    return - correct / predictions

def scoreFunction(predicted, correct):
    predicted_vec = n[0:, 0]
    correct_vec = n[0:, 0]
    return np.mean(predicted_vec == correct_vec)

##############################
# Neural model
##############################
class Model:
    def __init__(self, inputCnt, outputCnt, hiddenCnts, learnRate):
        layersSizes = [inputCnt] + hiddenCnts + [outputCnt] 
        self.layers = []

        # Add all layers except the last one
        for i in range(len(layersSizes) - 1):
            self.layers.append(FullLayer(layersSizes[i], layersSizes[i+1], learnRate))
            if i == len(layersSizes) - 1:
                self.layers.append(OutputActivationFunc())
            else:
                self.layers.append(HiddenActivationFunc())

    @staticmethod
    def _union_shuffle(a, b):
        result_a = np.empty(a.shape)
        result_b = np.empty(b.shape)
        for old_idx, new_idx in enumerate(np.random.permutation(len(a))):
            result_a[new_idx], result_b[new_idx] = a[old_idx], b[new_idx]
        return result_a, result_b

    def train(self, input, output, epochs, batchSize, validation_split):
        results = []

        input, output = Model._union_shuffle(input, output)

        trainSize = int(len(input) * validation_split)
        xTrain, yTrain = input[:trainSize], output[:trainSize]
        xValid, yValid = input[trainSize:], output[trainSize:]

        for _ in range(epochs):
            xTrain, yTrain = Model._union_shuffle(xTrain, yTrain)

            xParts = [xTrain[i : i + batchSize] for i in range(0, len(xTrain), batchSize)]
            yParts = [yTrain[i : i + batchSize] for i in range(0, len(yTrain), batchSize)]
            
            eachPartLoss = []

            for idx, part in enumerate(xParts):
                preds = self.predict(part)

                partLoss = lossFunction(preds, yParts[idx])
                eachPartLoss.append(partLoss)

                grad = lossGradient(preds, yParts[idx])
                for layer in reversed(self.layers):
                    grad = layer.back(grad)
            
            results.append({
                'loss': np.mean(eachPartLoss),
                'score': self.score(xValid, yValid),
            })
        return results

    def predict(self, input):
        for layer in self.layers:
            input = layer.feed(input)
        return input

    def verify(self, input, correct):
        preds = self.predict(input)
        return lossFunction(preds, correct)
    
    def score(self, input, correct):
        preds = self.predict(input)
        return scoreFunction(preds, correct)
    

##############################
# Data load and entry point
##############################
def main():
    parser = argparse.ArgumentParser(description='Perceptron classification')
    parser.add_argument('--file_path', type=str, default='dane.csv', help='Path to training set')
    parser.add_argument('--delimiter', type=str, default=';', help='Path to training set')
    parser.add_argument('--learn_rate', type=float, default=0.9, help='Learn rate')
    parser.add_argument('--neurons', type=int, nargs='+', help='Neurons configuration')
    args = parser.parse_args()

    info(f"Loading data from {args.file_path}")
    dataset = pandas.read_csv(args.file_path, args.delimiter)

    info("Preview of data:")
    print(dataset.sample(2))
    info("Data file format info:")
    print(dataset.info())

    xData = dataset.values[0:, :len(dataset.columns) - 1]
    yData = dataset.values[0:, len(dataset.columns) - 1]

    info("Setting up model")
    model = Model(len(xData), len(yData), args.neurons, args.learn_rate)

    info("Starting training")
    results = model.train(xData, yData, 1000, 100, 0.1)

if __name__ == "__main__":
    main()
