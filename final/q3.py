import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt

data = {'num_clusters': [1,2,3,4,5,6,7,8,9,10],
        'Elbow': [579857.9543, 532455.2722, 493218.0813, 433215.8150, 430290.4574, 412804.9312, 409729.7423,
                 404285.7518, 378087.1355, 369686.6227],
       'Silhouette': [0, 0.5391, 0.5300, 0.5479, 0.5411, 0.5140, 0.5172, 0.5081, 0.5056, 0.4984]}

clustering = pd.DataFrame(data)
clustering['slope'] = 0.0
clustering['acceleration'] = 0.0

for i in range(1, len(clustering)):
    clustering['slope'].iloc[i] = clustering['Elbow'].iloc[i] - clustering['Elbow'].iloc[i-1]

for i in range(2, len(clustering)):
    clustering['acceleration'].iloc[i] = clustering['slope'].iloc[i] - clustering['slope'].iloc[i-1]

sorted_clustering = clustering.sort_values(by=['acceleration'], ascending=False)
# print(clustering)
sorted_clustering

plt.plot(clustering['num_clusters'], clustering['Elbow'], linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(np.arange(2, 11, step = 1))
plt.show()

plt.plot(clustering['num_clusters'], clustering['Silhouette'], linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette")
plt.xticks(np.arange(2, 11, step = 1))
plt.show()