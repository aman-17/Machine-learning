"""
@Name: Assignment1_Q3.py
@Creation Date: February 3, 2023
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import random
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 7, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7}'.format

from sklearn.neighbors import KNeighborsClassifier

def simpleRandomSample (obsIndex, trainFraction = 0.7):
   '''Generate a simple random sample

   Parameters
   ----------
   obsIndex - a list of indices to the observations
   trainFraction - the fraction of observations assigned to Training partition
                   (a value between 0 and 1)

   Output
   ------
   trainIndex - a list of indices of original observations assigned to Training partition
   '''

   trainIndex = []

   nPopulation = len(obsIndex)
   nSample = round(trainFraction * nPopulation)
   kObs = 0
   iSample = 0
   for oi in obsIndex:
      kObs = kObs + 1
      U = random.random()
      uThreshold = (nSample - iSample) / (nPopulation - kObs + 1)
      if (U < uThreshold):
         trainIndex.append(oi)
         iSample = iSample + 1

      if (iSample == nSample):
         break

   testIndex = list(set(obsIndex) - set(trainIndex))
   return (trainIndex, testIndex)

fraud_data = pandas.read_csv('./AS1/Fraud.csv')

label = 'FRAUD'

feature_list = ['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']
n_feature = len(feature_list)

# Part (a)

fraud_frequency = fraud_data['FRAUD'].value_counts()
print(fraud_frequency)

fraud_rate = fraud_frequency.loc[1] / fraud_frequency.sum()
fraud_rate = round(fraud_rate, 4)

# Part (b)

sample_data = fraud_data[[label] + feature_list].dropna()

# Find the observed strata
strata = sample_data.groupby(label).count()

train_index = []
test_index = []

random.seed(a = 20230225)

for stratum in strata.index:
   obsIndex = sample_data[sample_data[label] == stratum].index
   trIndex, ttIndex = simpleRandomSample (obsIndex, trainFraction = 0.8)
   train_index.extend(trIndex)
   test_index.extend(ttIndex)
   
X_train = sample_data.loc[train_index][feature_list]
y_train = sample_data.loc[train_index][label]

label_freq_train = y_train.value_counts()
summary_stat_train = X_train[feature_list].describe()

X_test = sample_data.loc[test_index][feature_list]
y_test = sample_data.loc[test_index][label]

label_freq_test = y_test.value_counts()
summary_stat_test = X_test[feature_list].describe()

# Part (c)

result_list = []

# Determine the optimal number of nearest neighbors
for n_neighbors in range(2,8):
   model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'euclidean')
   nbrs = model.fit(X_train, y_train)

   pred_prob_train = nbrs.predict_proba(X_train)
   pred_y_train = numpy.where(pred_prob_train[:,1] >= fraud_rate, 1, 0)
   mce_train = numpy.mean(numpy.where(y_train.to_numpy() == pred_y_train, 0, 1))

   pred_prob_test = nbrs.predict_proba(X_test)
   pred_y_test = numpy.where(pred_prob_test[:,1] >= fraud_rate, 1, 0)
   mce_test = numpy.mean(numpy.where(y_test.to_numpy() == pred_y_test, 0, 1))

   result_list.append([n_neighbors, mce_train, mce_test])

result_df = pandas.DataFrame(result_list, columns = ['N Neighbors', 'MCE Training', 'MCE Testing'])

# Part (d)

plt.figure(figsize = (8,4), dpi = 200)
plt.plot(result_df['N Neighbors'], result_df['MCE Training'], color = 'green', marker = '^', label = 'Training')
plt.plot(result_df['N Neighbors'], result_df['MCE Testing'], color = 'royalblue', marker = 'o', label = 'Testing')
plt.xlabel('Number of Neighbors')
plt.ylabel('Misclassification Rate')
plt.xticks(range(2,8,1))
plt.yticks(numpy.arange(0.0, 0.7, 0.1))
plt.grid(axis = 'both')
plt.legend()
# plt.show()

# Part (e)

ipos = result_df['MCE Testing'].argmin()
n_neighbors = result_df.iloc[ipos]['N Neighbors'].astype(int)

model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'euclidean')
nbrs = model.fit(X_train, y_train)

focal = pandas.DataFrame(index = [0], columns = X_test.columns)
focal.loc[0]['DOCTOR_VISITS'] = 8
focal.loc[0]['MEMBER_DURATION'] = 178
focal.loc[0]['NUM_CLAIMS'] = 0
focal.loc[0]['NUM_MEMBERS'] = 2
focal.loc[0]['OPTOM_PRESC'] = 1
focal.loc[0]['TOTAL_SPEND'] = 16300

# neigh_focal contains the positions of the neighbors in the array, not the indices
dist_focal, neigh_focal = nbrs.kneighbors(focal)


# Retrieve the neighbors
X_subset = X_train.iloc[neigh_focal[0]]
y_subset = y_train.iloc[neigh_focal[0]]

# print(y_subset)
# Predicted probability
pred_prob_focal = nbrs.predict_proba(focal)
pred_y_focal = numpy.where(pred_prob_focal[:,1] >= fraud_rate, 1, 0)