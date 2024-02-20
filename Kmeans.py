import pandas as pd
import numpy as np
import matplotlib.pyplot as mplt

# data reading
DataSet = pd.read_csv("MidtermProject-part2-data.csv")

# finds euclidean distance between two points
def EuclideanDistance(centroid, point):
    distance = np.linalg.norm(centroid - point)
    return distance
#distance = math.sqrt(sum(pow(centroid - point, 2)))

# finds nearest centroid for the point
def FindNearestCentroid(centroidArr, point):
    nearestDistance = 100 
    closestIndex = 0

    for index, centroid in enumerate(centroidArr):
        distance = EuclideanDistance(centroid, point)
        if(distance < nearestDistance):
                    nearestDistance = distance
                    closestIndex = index
    return closestIndex                


# puts points to correct clusters by looking at their nearest centroid
def PutToCluster(centroidArr, pointArr):
    clusters = {}
    for centroidIndex in range(len(centroidArr)):
         clusters[centroidIndex] = []
    
    for pointIndex, point in enumerate(pointArr):
         closestIndex = FindNearestCentroid(centroidArr, point)
         clusters[closestIndex].append(point)

    return clusters     
            
 
# finds the mean of the cluster
def MeanOfCluster(cluster):
     sum = 0
     for point in cluster:
          sum += point
     mean = sum / len(cluster)
     return mean 


#creantes new centroids
def createNewCentroids(clusters):
     centroidArr = []
     for cluster in clusters.values():
        centroid = MeanOfCluster(cluster) #mean of that cluster chosen as the new centroid
        centroidArr.append(centroid)
     return centroidArr      


# data normalizing
def NormalizeData(data):
     return (data - data.min()) / (data.max() - data.min())


# K-Means algorithm
def Kmeans(k, epochs, dataSet):
     centroidArr = dataSet.sample(k).to_numpy()
     pointArr = dataSet.values

     for epoch in range(epochs):
          clusters = PutToCluster(centroidArr, pointArr)
          centroidArr = createNewCentroids(clusters)
     return pointArr, centroidArr, clusters             


# calculating wcss
def CalculateWCSS(clusters, centroidArr):
     wcss = 0
     for centroidIndex, centroid in enumerate(centroidArr):
          for pointIndex, point in enumerate(clusters[centroidIndex]): 
               wcss += pow(EuclideanDistance(centroid, point), 2)
     return wcss


#calculating bcss
def CalculateBCSS(centroidArr, clusters):
     bcss = 0
     centroidMidpoint = sum(centroidArr) / len(centroidArr)
     for centroidIndex,centroid in enumerate(centroidArr):
          bcss += pow(EuclideanDistance(centroidMidpoint, centroid), 2) * len(clusters[centroidIndex])   
     return bcss

# calculating dunn index
def DunnIndex(centroidArr, pointArr):
     dunnIndex = 0
     lowestInterCluster = 100
     highestIntraCluster = 0
     interCluster = 0
     intraCluster = 0
     # finds the smallest distance between any two cluster centroids
     for centroidIndexOut,centroidOut in enumerate(centroidArr):
          for centroidIndex,centroid in enumerate(centroidArr):
               if not np.allclose(centroid, centroidOut):
                    interCluster = EuclideanDistance(centroid, centroidOut)
                    if interCluster < lowestInterCluster:
                         lowestInterCluster = interCluster
     # finds the largest distance between any two points in any cluster                   
     for pointIndexOut, pointOut in enumerate(pointArr):
          for pointIndex, point in enumerate(pointArr):
               intraCluster = EuclideanDistance(point, pointOut)
               if(intraCluster > highestIntraCluster):
                    highestIntraCluster = intraCluster
     dunnIndex =  lowestInterCluster / highestIntraCluster               
     return dunnIndex


# shows calculation results
def dataResult(pointArr, centroidArr, clusters):
    print("Wcss: ", CalculateWCSS(clusters, centroidArr))
    print("Bcss: ", CalculateBCSS(centroidArr, clusters))
    print("Dunn Index: ", DunnIndex(centroidArr, pointArr))






