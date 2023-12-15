import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as function
import sklearn.metrics as metrics

# data reading
TrainSet = pd.read_csv("midtermProject-part1-TRAIN.csv")
DataResults = TrainSet.values[:, -1]

TensorValues = torch.Tensor(TrainSet.values[:, :-1])
TensorResults = torch.Tensor(TrainSet.values[:, -1])

TestSet = pd.read_csv("midtermProject-part1-TEST.csv")
TestSetResults = TestSet.values[:, -1]
TestTensorValues = torch.Tensor(TestSet.values[:, :-1])
TestTensorResults = torch.Tensor(TestSet.values[:, -1])


# creates a neural network model
class Model(nn.Module):
        # initializes the neural network model
        def __init__(self, hiddenLayerNodeCount: np.array, inputFeatures=32, outFeatures = 1):
            super().__init__()
            self.layerconnections = nn.ModuleList()
            # connection between input and first hidden layer
            self.layerconnections.append(nn.Linear(inputFeatures, hiddenLayerNodeCount[0])) 
            # connections between hidden layers
            for hiddenLayerIndex,_ in enumerate(hiddenLayerNodeCount): 
                if hiddenLayerIndex <  len(hiddenLayerNodeCount)-1:  
                        self.layerconnections.append(nn.Linear(hiddenLayerNodeCount[hiddenLayerIndex], hiddenLayerNodeCount[hiddenLayerIndex+1]))
            # connection between last hidden layer and output               
            self.layerconnections.append(nn.Linear(hiddenLayerNodeCount[hiddenLayerIndex], outFeatures))    

        # feedforwarding with using relu activation function   
        def feedforward(self, data):
            for layerIndex,layer in enumerate(self.layerconnections):
                if layerIndex < len(self.layerconnections)-1:
                    # using relu activation function
                    data = function.relu(layer(data)) 
                else:
                    data = layer(data)       
            return data
        
        criterion = nn.MSELoss()


# shows results           
def showResults(actual, prediction):
    mae = metrics.mean_absolute_error(actual, prediction)
    mse = metrics.mean_squared_error(actual, prediction)
    rmse = np.sqrt(mse)  
    r2 = metrics.r2_score(actual,prediction)


    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R^2 (Coefficient of determination): ", r2)


