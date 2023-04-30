import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import time
from sklearn.neural_network import MLPRegressor
import Utility
from itertools import combinations
from sklearn.metrics import accuracy_score

catName = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', \
           'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
nPredictor = len(catName)
inputData = pandas.read_excel('./AS4/Homeowner_Claim_History.xlsx',
                              sheet_name = 'HOCLAIMDATA')
yName = 'claim_group'
# inputData[yName] = numpy.where(inputData['exposure'] > 0.0, inputData['num_claims'] / inputData['amt_claims'], numpy.NaN)

inputData['claim_group'] = pandas.cut(inputData['num_claims'], bins=[-1, 0, 1, 2, numpy.inf], labels=[0, 1, 2, 3])
inputData.drop('num_claims', axis=1, inplace=True)

trainData0 = inputData[inputData['policy'].str.startswith(('A', 'G', 'Z'))]
testData0 = inputData[~inputData['policy'].str.startswith(('A', 'G', 'Z'))]

trainData0 = trainData0[catName + [yName]].dropna().reset_index(drop = True)
testData0 = testData0[catName + [yName]].dropna().reset_index(drop = True)

allCombResult = []
allFeature = catName

allComb = []
for r in range(nPredictor+1):
   allComb = allComb + list(combinations(allFeature, r))

startTime = time.time()
maxIter = 20
tolS = 1e-7

nComb = len(allComb)
for r in range(nComb):
   modelTerm = list(allComb[r])
   trainData = trainData0[[yName] + modelTerm].dropna()
   n_sample = trainData.shape[0]
   X_train = trainData[[yName]].copy()
   X_train.insert(0, 'Intercept', 1.0)
   X_train.drop(columns = [yName], inplace = True)
   y_train = trainData[yName].copy()
   for pred in modelTerm:
      if (pred in catName):
         X_train = X_train.join(pandas.get_dummies(trainData[pred].astype('category')))

   modelTerm = list(allComb[r])
   testData = testData0[[yName] + modelTerm].dropna()
   n_sample_test = testData.shape[0]
   X_test = testData[[yName]].copy()
   X_test.insert(0, 'Intercept', 1.0)
   X_test.drop(columns = [yName], inplace = True)
   y_test = testData[yName].copy()
   for pred in modelTerm:
      if (pred in catName):
         X_test = X_test.join(pandas.get_dummies(testData[pred].astype('category')))
   n_param = X_test.shape[1]
   XtX = X_test.transpose().dot(X_test)
   origDiag = numpy.diag(XtX)
   XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = tolS)
   X_reduce = X_test.iloc[:, list(nonAliasParam)]
   resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
   modelFit = resultList[0]
   y_pred = modelFit.predict(X_reduce)
   y_simple_residual = y_test - y_pred
   y_pred = y_pred.idxmax(axis=1)
   accuracy = accuracy_score(y_test, y_pred)
   mse = numpy.mean(numpy.power(y_simple_residual, 2))
   rmse = numpy.sqrt(mse)
   modelLLK = resultList[1]
   modelDF = resultList[2]
   del resultList

   AIC = 2.0 * modelDF - 2.0 * modelLLK
   BIC = modelDF * numpy.log(n_sample) - 2.0 * modelLLK
   allCombResult.append([r, modelTerm, len(modelTerm), modelLLK, modelDF, AIC, BIC, n_sample, rmse, accuracy])

endTime = time.time()

allCombResult = pandas.DataFrame(allCombResult, columns = ['Step', 'Model Term', 'Number of Terms', \
                'Log-Likelihood', 'Model Degree of Freedom', 'Akaike Information Criterion', \
                'Bayesian Information Criterion', 'Sample Size', 'Test_RMSE', 'Test_Accuracy'])

allCombResult.to_csv('as5_q1.csv')
