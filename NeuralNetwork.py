import pandas as pd
import numpy as np
import math
import random

TrainSet = pd.read_csv("midtermProject-part1-TRAIN.csv")
DataResults = TrainSet.values[:, -1]

#print(DataResults)




def CalculateWeight():
    return

def CalculateBias():
    return

def Softmax(x):
    return (math.log(1+ math.exp(x)))

def ReLU(x):
    return max(0, x)

def Sigmoid(x):
    return (math.exp(x)) / (math.exp(x) + 1)

def Backpropagation():
    return

def Epoch():
    return


