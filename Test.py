from NeuralNetwork import *

print("\nResults for train set: ")
model = Model([6,5])
#print(model)

print("The final values of the weights and biases: ")
for name, param in model.named_parameters():
    print(name, param)
      

# optimazing
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# data training 
def TrainData():
     epochs = 200
     losses = []
     
     for epoch in range(epochs):
          prediction = model.feedforward(TensorValues)
          corretResults = TensorResults.reshape(-1, 1)
          loss = model.criterion(prediction, corretResults)
          losses.append(loss.detach().numpy())
    
          #backpropagation
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if epoch % 10 == 0:
           print(f'Epoch: {epoch} and loss: {loss}')

     # shows training results
     showResults(DataResults, prediction.detach().numpy())


TrainData()  


print("\nResults for test set: ")

# preventing backpropagation for testing
with torch.no_grad():
      testPrediction = model.feedforward(TestTensorValues)
      # shows test results
      showResults(TestSetResults, testPrediction.detach().numpy())

