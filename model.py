#!/usr/bin/env python3
import argparse
import numpy as np
import pandas
import random
import sys

from scipy import stats


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
        center = 0
        lower, upper = -1, 1
        sigma = 1

        norm_dist = stats.truncnorm((lower - center) / sigma, (upper - center) / sigma, loc=center, scale=sigma)
        self.weights = np.matrix([
            [norm_dist.rvs() for _ in range(outputCnt)] for _ in range(inputCnt)
        ])

        # create vector of random biases
        self.biases = np.array([norm_dist.rvs() for _ in range(outputCnt)])
        self.learnRate = learnRate

    def feed(self, input):
        self.old_input = input
        return np.dot(input, self.weights) + self.biases

    def back(self, output):
        old_input = self.old_input.T
        grad = np.dot(output, self.weights.T)

        grad_weights = np.dot(old_input, output)
        self.weights = self.weights - self.learnRate * grad_weights

        grad_biases = output.mean(axis=0) * self.old_input.shape[0]
        self.biases = self.biases - self.learnRate * grad_biases

        return grad

# Leaky reLU
class HiddenActivationFunc:    
    def feed(self, input):
        self.old_input = input
        return np.maximum(0.01 * input, input)
    
    def back(self, output):
        return np.asarray(output) * np.asarray(np.where(self.old_input > 0, 1, 0.01))

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

# Use custom epsilon (instead of sys.float_info.epsilon) to avoid awful math errors
#  caused by limited precision of floats
bigger_epsilon = 1e-6

def lossFunction(predicted, correct):
    predictions = np.clip(predicted, bigger_epsilon, 1.0 - bigger_epsilon)
    size = predictions.shape[0]
    return -np.sum(correct * np.log(predictions) + (1-correct) * np.log(1 - predictions)) / size

def lossGradient(predicted, correct):
    predictions = np.clip(predicted, bigger_epsilon, 1.0 - bigger_epsilon)
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
            loss_on_val = self.verify(xValid, yValid)

            self.TF(xValid, yValid)
            results.append({
                'loss': loss,
                'score': score,
                'loss_on_val': loss_on_val,
            })

            info(f"\t Done {currentEpoch}/{epochs} - score ({score}), loss ({loss}), loss_on_val ({loss_on_val})")
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
    
    def TF(self, input, correct):
        preds = self.predict(input)
        predicted_vec = preds[0:, 0]
        TP, TN, FP, FN = 0,0,0,0
        for i in range(len(predicted_vec)):
            if predicted_vec[i] > 0.5 and correct[i] == 1:
                TP += 1
            elif predicted_vec[i] > 0.5 and correct[i] == 0:
                FP += 1
            elif predicted_vec[i] <= 0.5 and correct[i] == 0:
                TN += 1
            elif predicted_vec[i] <= 0.5 and correct[i] == 1:
                FN += 1
        info('\tTP: {}, FP: {}, TN: {}, FN: {}'.format(TP,FP,TN,FN))
        return
    

##############################
# Data load and entry point
##############################
# Useful constants for debugging
DATA_SIZE = -1              # -1 for full
VALIDATION_SPLIT = 0.7

def main():
    parser = argparse.ArgumentParser(description='Perceptron classification')
    parser.add_argument('--file_path', type=str, default='dane.csv', help='Path to training set')
    parser.add_argument('--learn_rate', type=float, default=0.1, help='Learn rate')
    parser.add_argument('--neurons', type=int, nargs='+', default=[], help='Neurons configuration')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs used in training')
    parser.add_argument('--batch_size', type=int, default=5, help='How often to recalculate gradient')
    args = parser.parse_args()

    info(f"Loading data from {args.file_path}")
    dataset = pandas.read_csv(args.file_path, ";")

    info("Preview of data:")
    print(dataset.sample(2))
    info("Data file format info:")
    print(dataset.info())

    xSize, ySize = len(dataset.columns) - 1, 1
    xData = dataset.values[0:DATA_SIZE, :len(dataset.columns) - 1]
    yData = dataset.values[0:DATA_SIZE, len(dataset.columns) - 1]

    # Normalize input data
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
    nn = Model(xSize, ySize, args.neurons, args.learn_rate)

    info("Training...")
    info("(This might take a while - better get yourself a beer)")
    results = nn.train(newxData, yData, args.epochs, args.batch_size, VALIDATION_SPLIT)

    info("Dumping to file...")
    with open("results.csv", "w") as f:
        for i, line in enumerate(results):
            if i == 0:
                f.write(";".join(line.keys()))
                f.write('\n')
            f.write(";".join([str(x) for x in line.values()]))
            f.write('\n')

    info("Finished. Have a nice day!")

if __name__ == "__main__":
    main()
