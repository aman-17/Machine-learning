import os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

dataset = pd.read_csv('./AS2/TwoFeatures.csv', delimiter=',')
features_data = ['x1','x2']

def manhattanDistance(dataset, clusterNum)->float:
   center = dataset.to_numpy()[range(clusterNum),:]
   for iter in itertools.islice(itertools.count(), 500):
      distance = metrics.pairwise.manhattan_distances(dataset.to_numpy(), center)
      center = np.full((clusterNum, dataset.shape[1]), 0.0)
      for cluster in itertools.islice(itertools.count(), clusterNum):
         inCluster = (np.argmin(distance, axis = 1) == cluster)
         if (np.sum(inCluster) > 0):
            center[cluster,:] = np.mean(dataset.to_numpy()[inCluster,], axis = 0)
      member_diff = np.sum(np.abs(np.argmin(distance, axis = 1) - np.zeros(dataset.shape[0])))
      if (member_diff > 0):
          member_prev = np.argmin(distance, axis = 1)
      else:
          break
   return np.argmin(distance, axis = 1), np.min(distance, axis = 1), center

def wcss(data)->pd.DataFrame:
   data_new = pd.DataFrame()
   for k in itertools.islice(itertools.count(), 8):
      clusterNumbers = k + 1
      argmin_manhattan, min_manhattanDist, center  = manhattanDistance(data, clusterNumbers)[0], manhattanDistance(data, clusterNumbers)[1], manhattanDistance(data, clusterNumbers)[2]
      withinClusterS = np.zeros(clusterNumbers)
      numberClusters = np.zeros(clusterNumbers)
      withinClusterST = 0.0
      optimalWCS = 0.0
      for k in itertools.islice(itertools.count(), clusterNumbers):
         numberClusters[k] = np.sum(np.where(argmin_manhattan == k, 1, 0))
         withinClusterS[k] = np.sum(np.where(argmin_manhattan == k, min_manhattanDist **2, 0.0))
         optimalWCS += withinClusterS[k] / numberClusters[k]
         withinClusterST += withinClusterS[k]
      print(f"centroid for k: {clusterNumbers} is {center}.")
      data_new = data_new.append([[clusterNumbers, withinClusterST, optimalWCS, center]], ignore_index = True)
   data_new.columns = ['No. of Clusters', 'TWCSS', 'Elbow Value', 'centroid']
   # print(data_new)
   return data_new

def centroid(clusters, scale = 10)->None:
   feature_minimum = dataset[features_data].min()
   feature_maximum = dataset[features_data].max()
   dataset[features_data] = 10.0 * (dataset[features_data] - feature_minimum) / (feature_maximum - feature_minimum)
   center = manhattanDistance(dataset, clusters)[2]
   arange = feature_maximum - feature_minimum
   centroid_oscale = center / scale
   for j in itertools.islice(itertools.count(), 2):
      centroid_oscale[:,j] = centroid_oscale[:,j] * arange[j] + feature_minimum[j]
   print(centroid_oscale)
   return None

# a. Plot x2 (vertical axis) versus x1 (horizontal axis).  Add gridlines to both axes.  Let the graph engine chooses the tick marks. How many clusters do you see in the graph?
plt.figure(dpi = 100)
plt.scatter(dataset['x1'], dataset['x2'], marker = '*', color = 'black')
plt.title('Features')
plt.grid(axis='both', linestyle = ':')
plt.xlabel('x1')
plt.ylabel('x2')
plt.margins(0.15)
plt.show()

# b. Discover the optimal number of clusters without any transformations.  List the number of clusters, the Total Within-Cluster Sum of Squares (TWCSS), and the Elbow values in a table. Plot the Elbow Values versus the number of clusters.  How many clusters do you find? What are the centroids of your optimal clusters?
os_metric = wcss(dataset)

plt.figure(dpi = 100)
plt.plot(os_metric['No. of Clusters'], os_metric['Elbow Value'], linewidth = 2, marker = 'o', color = 'black')
plt.title('Features in Original Scale')
plt.grid(axis='both', linestyle = ':')
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.margins(y = 0.15)
plt.xticks(os_metric['No. of Clusters'])
plt.show()
centroid(clusters = 2)


# c. Linearly rescale x1 such that the resulting variable has a minimum of zero and a maximum of ten.  Likewise, rescale x2.  Discover the optimal number of clusters from the transformed observations. List the number of clusters, the Total Within-Cluster Sum of Squares (TWCSS), and the Elbow values in a table. Plot the Elbow Values versus the number of clusters.  How many clusters do you find? What are the centroids of your optimal clusters in the original scale of x1 and x2?
scl_metric = wcss (dataset)
plt.figure(dpi = 100)
plt.plot(scl_metric['No. of Clusters'], scl_metric['Elbow Value'], linewidth = 2, marker = 'o', color = 'black')
plt.title('Features in Transformed Scales')
plt.grid(axis='both', linestyle = ':')
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.margins(y = 0.15)
plt.xticks(scl_metric['No. of Clusters'])
plt.show()
centroid(clusters = 4)