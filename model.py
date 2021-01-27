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
            [random.uniform(0, 1) for _ in range(outputCnt)] for _ in range(inputCnt)
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
        return np.asarray(output) * np.asarray(self.old_input > 0)

# Sigmoid
class OutputActivationFunc:
    def feed(self, input):
        input = np.clip(input, -10, 10)
        self.old_input = input
        return 1/(1+np.exp(-input))
    
    def back(self, output):
        old_input = np.asarray(self.old_input).reshape(-1)
        old_input = np.clip(old_input, -10, 10)
        result = output * (1/(1+np.exp(-old_input))) * (1 - 1/(1+np.exp(-old_input)))
        return np.asmatrix(result).T

##############################
# Score functions
##############################
def lossFunction(predicted, correct):
    predictions = np.clip(predicted, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
    batch_size = predictions.shape[0]
    ce = -np.sum(correct * np.log(predictions) + (1-correct) * np.log(1 - predictions)) / batch_size
    return ce

def lossGradient(predicted, correct):
    predictions = np.clip(predicted, sys.float_info.epsilon, 1.0 - sys.float_info.epsilon)
    predictions = np.asarray(predictions).reshape(-1)
    return - correct / predictions + (1-correct) / (1 - predictions)

def scoreFunction(predicted, correct):
    predicted_vec = predicted[0:, 0]
    return np.mean((predicted_vec > 0.5) == correct)

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
            if i == len(layersSizes) - 2:
                self.layers.append(OutputActivationFunc())
            else:
                self.layers.append(HiddenActivationFunc())

    @staticmethod
    def _union_shuffle(a, b):
        result_a = np.empty(a.shape, dtype=a.dtype)
        result_b = np.empty(b.shape, dtype=b.dtype)
        for old_idx, new_idx in enumerate(np.random.permutation(len(a))):
            result_a[new_idx], result_b[new_idx] = a[old_idx], b[old_idx]
        return result_a, result_b

    def train(self, input, output, epochs, batchSize, validation_split):
        results = []

        input, output = Model._union_shuffle(input, output)

        trainSize = int(len(input) * validation_split)
        xTrain, yTrain = input[:trainSize], output[:trainSize]
        xValid, yValid = input[trainSize:], output[trainSize:]

        for currentEpoch in range(epochs):
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
            
            loss = np.mean(eachPartLoss)
            score = self.score(xValid, yValid)
            print(self.score(xTrain, yTrain))
            info(f"\t Done {currentEpoch}/{epochs} - score ({score}), loss ({loss})")
            results.append({
                'loss': loss,
                'score': score,
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
    parser.add_argument('--learn_rate', type=float, default=0.1, help='Learn rate')
    parser.add_argument('--neurons', type=int, nargs='+', default=[], help='Neurons configuration')
    args = parser.parse_args()

    info(f"Loading data from {args.file_path}")
    dataset = pandas.read_csv(args.file_path, args.delimiter)

    info("Preview of data:")
    print(dataset.sample(2))
    info("Data file format info:")
    print(dataset.info())

    DATA_SIZE = 1000      # -1 for full

    xSize, ySize = len(dataset.columns) - 1, 1
    xData = dataset.values[0:DATA_SIZE, :len(dataset.columns) - 1]
    yData = dataset.values[0:DATA_SIZE, len(dataset.columns) - 1]

    newxData = []
    for i in range(xSize):
        row = xData[0:, i]
        maxVal = np.amax(row)
        minVal = np.amin(row)
        diff = maxVal - minVal
        newRow = np.empty(len(row))
        for j in range(len(newRow)):
            newRow[j] = float(row[j] - minVal) / float(diff)
        newxData += [newRow]
    newxData = np.array(newxData).T


    info("Setting up model")
    print(xSize, ySize, len(xData), len(yData))
    model = Model(xSize, ySize, args.neurons, args.learn_rate)

    info("Training...")
    info("(This might take a while - better get yourself a beer)")
    results = model.train(newxData, yData, 100, 2, 0.8)

if __name__ == "__main__":
    main()
