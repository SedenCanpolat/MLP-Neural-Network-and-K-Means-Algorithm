import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt

# visualizates the data 
def visualization(k, clusters):
    #taking input from the user as column selection
    col1 = int(input("Select your first column between 0-8: "))
    col2 = int(input("Select your second column between 0-8: "))

    for i in range(k):
        x = np.array(clusters[i])[:,col1]
        y = np.array(clusters[i])[:,col2]
        mplt.scatter(x, y)
    #naming    
    mplt.xlabel('Feature 1')
    mplt.ylabel('Feature 2')    
    mplt.show()


