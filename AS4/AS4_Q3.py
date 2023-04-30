import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import time
from sklearn.neural_network import MLPRegressor
import Utility

catName = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', \
           'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
nPredictor = len(catName)
inputData = pandas.read_excel('./AS4/Homeowner_Claim_History.xlsx',
                              sheet_name = 'HOCLAIMDATA')
yName = 'Frequency'
inputData[yName] = numpy.where(inputData['exposure'] > 0.0, inputData['num_claims'] / inputData['amt_claims'], numpy.NaN)

trainData = inputData[inputData['policy'].str.startswith(('A', 'G', 'Z'))]
testData = inputData[~inputData['policy'].str.startswith(('A', 'G', 'Z'))]

trainData = trainData[catName + [yName]].dropna().reset_index(drop = True)
testData = testData[catName + [yName]].dropna().reset_index(drop = True)

print(len(trainData))
print(len(testData))

for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

for pred in catName:
    u = testData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    testData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

X0 = pandas.get_dummies(trainData[catName].astype('category'))
X0.insert(0, '_BIAS_', 1.0)

n_param = X0.shape[1]
XtX = X0.transpose().dot(X0)
origDiag = numpy.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator(n_param, XtX, origDiag, sweepCol = range(n_param), tol = 1.0e-7)
X_train = X0.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_'])
y_train = trainData[yName].copy()

X1 = pandas.get_dummies(testData[catName].astype('category'))
X1.insert(0, '_BIAS_', 1.0)

n_param1 = X1.shape[1]
XtX1 = X1.transpose().dot(X1)
origDiag1 = numpy.diag(XtX1)
XtXGinv1, aliasParam1, nonAliasParam1 = Utility.SWEEPOperator(n_param1, XtX1, origDiag1, sweepCol = range(n_param1), tol = 1.0e-7)
X_test = X1.iloc[:, list(nonAliasParam1)].drop(columns = ['_BIAS_'])
y_test = testData[yName].copy()

result = pandas.DataFrame()
actFunc = ['tanh', 'identity', 'relu']
nLayer = [1,2,3,4,5,6,7,8,9,10]
nHiddenNeuron = [1,2,3,4,5]

maxIter = 10000
randSeed = 2023484

combList = itertools.product(actFunc, nLayer, nHiddenNeuron)
result = []
mse0 = numpy.var(y_test, ddof = 0)

for comb in combList:
   print(comb)
   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   nHiddenNeuron = comb[2]
   nnObj = MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer, \
                        activation = actFunc, verbose = False, \
                        max_iter = maxIter, random_state = randSeed)
   thisFit = nnObj.fit(X_train, y_train)
   y_pred = thisFit.predict(X_test)

   y_simple_residual = y_test - y_pred
   mse = numpy.mean(numpy.power(y_simple_residual, 2))
   rmse = numpy.sqrt(mse)
   relerr = mse / mse0
   corr_matrix = numpy.corrcoef(y_test, y_pred)
   pearson_corr = corr_matrix[0,1]
   time_end = time.time()
   time_elapsed = time_end - time_begin
   result.append([actFunc, nLayer, nHiddenNeuron, thisFit.n_iter_, thisFit.best_loss_, \
                  rmse, relerr, pearson_corr, time_elapsed])

result_df = pandas.DataFrame(result, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', \
               'N Iteration', 'Loss', 'RMSE', 'RelErr', 'Pearson Corr', 'Time Elapsed'])

print(result_df)
result_df.to_csv('out.csv')
ipos = numpy.argmin(result_df['RMSE'])
row = result_df.iloc[ipos]

actFunc = row['Activation Function']
nLayer = row['nLayer']
nHiddenNeuron = row['nHiddenNeuron']

print('=== Optimal Model ===')
print('Activation Function: ', actFunc)
print('Number of Layers: ', nLayer)
print('Number of Neurons: ', nHiddenNeuron)
