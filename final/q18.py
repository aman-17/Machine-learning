import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as stats
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('./AS5/WineQuality_Train.csv')
test_data = pd.read_csv('./AS5/WineQuality_Test.csv')
threshold = 0.1961733010776
Xtrain = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtrain = stats.add_constant(Xtrain, prepend=True)
Ytrain = train_data['quality_grp']

Xtest = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtest = stats.add_constant(Xtest, prepend=True)
Ytest = test_data['quality_grp']
np.random.seed(0) 
n_iterations = 200 
misclassification_rates = []
for i in range(1, n_iterations + 1):
    Xtrain_resampled, Ytrain_resampled = resample(Xtrain, Ytrain)
    logit = stats.MNLogit(Ytrain_resampled, Xtrain_resampled)
    thisFit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8, disp=0)
    pred_prob = thisFit.predict(Xtest)
    predictions = [1 if p[1] >= threshold else 0 for p in pred_prob.values]
    misclassification_rate = 1 - accuracy_score(Ytest, predictions)
    misclassification_rates.append(misclassification_rate)

plt.plot(range(1, n_iterations + 1), misclassification_rates)
plt.xlabel('Number of Bagging Steps')
plt.ylabel('Misclassification Rate')
plt.show()