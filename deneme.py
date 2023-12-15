import pandas as pd
import numpy as np
import math
import random
import torch

TrainSet = pd.read_csv("midtermProject-part1-TRAIN.csv")
DataResults = TrainSet.values[:, -1]

#print(DataResults)

def CreateMatrix(rows, columns):
    matrix = []

    for i in range(rows):
        row = [0] * columns
        matrix.append(row)
    return matrix

#CreateMatrix(3,2)

def MatrixRandomize(rows, columns):
    matrix = CreateMatrix(rows, columns)
    for i in range(rows):
        for j in range(columns):
            matrix[i][j] = random.random()
    return print(matrix)

#MatrixRandomize(2,3)

def MatrixAdd(n, matrix):
    print(matrix)
    
    if(isinstance(n, np.ndarray)):
        if matrix.shape != n.shape:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    matrix[i][j] += n[i][j]
                    print('a')
        else:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    matrix[i][j] += n 
    print(matrix)                          
    return matrix


#MatrixAdd(np.matrix(2,1), np.matrix(3,5)) 

def MatrixMult(n, rows, columns):
    matrix = CreateMatrix(rows, columns)
    for i in range(rows):
        for j in range(columns):
            matrix[i][j] *= n
    print(matrix)

#MatrixMult(7, 3, 2)    



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


def feedforward(inputNodes, hiddenNodes, outputNodes):
    weightsInputHidden = np.matrix(hiddenNodes, inputNodes) #reverse?
    weightsHiddenOutput = np.matrix(outputNodes, hiddenNodes)
    weightsInputHiddenRandomized = np.random.rand(weightsInputHidden.shape[0], weightsInputHidden.shape[1]) * 2 - 1
    weightsHiddenOutputRandomized = np.random.rand(weightsHiddenOutput.shape[0], weightsHiddenOutput.shape[1]) * 2 - 1

    biasHidden =  np.matrix(hiddenNodes, 1)
    biasOutput = np.matrix(hiddenNodes, 1)
    
    hiddenNode = np.dot(weightsInputHiddenRandomized, inputNodes)
    hiddenNode += (biasHidden)

    return hiddenNode

def numpyarr():
    numpyarray = np.random.rand(2,3)
    print(numpyarray)
    tens = torch.tensor(numpyarray) #tensorfication
    print(tens)



def tensorcreate():
    tens = torch.randn(2,3)
    print(tens)


def tensReShape():
    tens = torch.randn(3,5)
    print(tens)
    #retens = tens.reshape(-1,3)
    #print(tens[1,:]) #row 1
    #print(tens[:, :-1]) #column till last one
    #print(tens[:, 1:]) #column starting from 1 to end


def aritmetic():
    tensorA = torch.tensor([1,2,3,4])
    tensorB = torch.tensor([5,6,7,8])

    sum = torch.add(tensorA, tensorB)
    mod = torch.remainder(tensorB, tensorA)
    power = torch.pow(tensorA, tensorB)
    tensorA.add_(tensorB) # A = B
    #tensorA = tensorA.add(tensorB)
    print(tensorA)

aritmetic()

def NeuralNetwork(inputNodes, hiddenNodes, outputNodes): #hiddenLayerCount, weights, bias):

    '''
    inputs = np.array(inputs)

    for i in range(hiddenLayerCount):
        hiddenLayer = (np.dot(weights, inputs) + bias)
        outputs = (np.dot(weights, hiddenLayer) + bias)
    print(hiddenLayer)
    return outputs

NeuralNetwork([0,1], 2, [2,3], 4)
    '''


inputNodes = np.array([1, 2, 3])  # Replace with your actual input
hiddenNodes = 4  # Replace with your actual number of hidden nodes
outputNodes = 1  # Replace with your actual number of output nodes

#result = feedforward(inputNodes, hiddenNodes, outputNodes)
#print(result)


 #prediction1D = prediction.unsqueeze(1)

 #def AssignNearestCentroid(centroids, points):

#print(EuclideanDistance(DataSet.sample(1), DataSet.values[1, :])) 

#def MeanOfCluster(cluster):
#   return sum(cluster)/len(cluster)
#clusters[min(range(len(clusters.values())), key=lambda i:abs(clusters[i]-mean))]

'''
if np.allclose(centroidArr, newcentroidArr):#np.array_equal(centroidArr, newcentroidArr):#centroidArr == newcentroidArr:
    print(clusters)

else:
    clusters = PutToCluster(centroidArr, pointArr)
'''

#distance = math.sqrt(sum(pow(centroid - point, 2)))