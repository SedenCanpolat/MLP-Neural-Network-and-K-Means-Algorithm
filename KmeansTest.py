from Kmeans import *
from DataVisualization import *

#data normalizing
DataSet = NormalizeData(DataSet)
pointArr, centroidArr, clusters = Kmeans(3, 100, DataSet)

dataResult(pointArr, centroidArr, clusters)


# writes the data to result.txt 

lines = ""
for index, point in enumerate(pointArr):
    for clusterIndex, cluster in enumerate(clusters):
        lines += f"\nRecord {index + 1}:"
        lines += f"\tCluster {clusterIndex + 1}"

lines += "\n"

# shows all clusters' record count
for clusterIndex, cluster in enumerate(clusters):
        lines += f"\nCluster {clusterIndex + 1}: {len(clusters[clusterIndex])} records"

# calculation results
lines += "\n"   
lines += f"\nWCSS: {CalculateWCSS (clusters, centroidArr)}"
lines += f"\nBCSS: {CalculateBCSS(centroidArr, clusters)}"
lines += f"\nDunn Index: {DunnIndex(centroidArr, clusters)}"

f = open("result.txt", "w")
f.writelines(lines)
f.close()

visualization(3, clusters)