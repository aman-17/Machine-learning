import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


X = np.array([[0.2,0],[0.3,0.6],[0.4,0],[0.5,0.2],[0.5,0.6],[0.5,0.6],[0.5,0.8],[0.7,0],[0.7,0.4],[0.8,0.6]])
Y = np.array([1.4,-1.0,1.8,1.8,0.4,0.4,1.2,2.4,1.0,2.0])


k = [i for i in range(2,10)]
error = []
for i in range(len(k)):
  mod = KNeighborsRegressor(n_neighbors = k[i], metric='chebyshev')
  mod.fit(X,Y)
  e = mean_absolute_error(Y,mod.predict(X))
  error.append(e)

plt.plot(k,error)
plt.xlabel('k values')
plt.ylabel('Absolute error')
plt.show()
print('Optimal k values : ',k[error.index(min(error))])