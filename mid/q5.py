import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = {
        'X':[0,0.15,0.35,0.4,0.55,0.6,0.55,0.7,0.8,0.85],
        'Y':[0.78,0.72,0.32,0.52,0.30,0.80,0.05,0.95,0.95,0.18]}

data1 = {
        'X':[0,0.30,0.70,0.80,1.10,1.2,1.10,1.4,1.6,1.70],
        'Y':[1.56,1.44,0.64,1.04,0.60,1.60,0.10,1.90,1.90,0.36]}


# X= np.random.rand(50,2)
# Y= 2 + np.random.rand(50,2)
# Z= np.concatenate((X,Y))
Z=pd.DataFrame(data) #converting into data frame for ease
print(Z.describe())

g = sns.scatterplot(Z['X'],Z['Y'])

KMean= KMeans(n_clusters=2)
KMean.fit(Z)
label=KMean.predict(Z)

print(f'Silhouette Score(n=1): {silhouette_score(Z, label)}')

modifiedZ = pd.DataFrame(data1)
print(modifiedZ.describe())

g1 = sns.scatterplot(modifiedZ['X'],modifiedZ['Y'])
sns.set_style({'axes.grid':True})


KMean= KMeans(n_clusters=2)
KMean.fit(modifiedZ)
label=KMean.predict(modifiedZ)

print(f'Silhouette Score(n=2): {silhouette_score(modifiedZ, label)}')