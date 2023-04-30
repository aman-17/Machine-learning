import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
hmeq = pd.read_csv('./AS1/hmeq.csv', delimiter=',', usecols=['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC'])

# a. Before we partition the observations, we need a baseline for reference.  How many observations are in the dataset?  What are the frequency distributions of BAD (including missing)? What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE?
print('Number of observations in the dataset are',len(hmeq))
print(hmeq.groupby('BAD').size())
print('Mean of DEBTINC: ', hmeq['DEBTINC'].mean())
print('Standard deviation of DEBTINC: ', hmeq['DEBTINC'].std())
print('Mean of LOAN: ', hmeq['LOAN'].mean())
print('Standard deviation of LOAN: ', hmeq['LOAN'].std())
print('Mean of MORTDUE: ', hmeq['MORTDUE'].mean())
print('Standard deviation of MORTDUE: ', hmeq['MORTDUE'].std())
print('Mean of VALUE: ', hmeq['VALUE'].mean())
print('Standard deviation of VALUE: ', hmeq['VALUE'].std())



# b. We first try the simple random sampling method. How many observations (including those with missing values in at least one variable) are in each partition?  What are the frequency distributions of BAD (including missing) in each partition? What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE in each partition?
def simpleRandomSample(obsIndex, trainFraction = 0.7):
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
   nSample = np.round(trainFraction * nPopulation)
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
def nbrs_metric (class_prob, y_class, n_class):
    nbrs_pred = np.argmax(class_prob, axis = 1)
    mce = np.mean(np.where(nbrs_pred == y_class, 0, 1))
    rase = 0.0
    for i in range(y_class.shape[0]):
        for j in range(n_class):
            if (y_class.iloc[i] == j):
                rase = rase + np.power((1.0 - class_prob[i,j]),2)
            else:
                rase = rase + np.power(class_prob[i,j],2)
    rase = np.sqrt(rase / y_class.shape[0] / n_class)
    return (mce, rase)   
sampleData = hmeq[['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
trainIndex = []
testIndex = []
obsIndex = sampleData.index
random.seed(a = 20230101)
trIndex, ttIndex = simpleRandomSample(obsIndex, trainFraction = 0.7)
trainIndex.extend(trIndex)
testIndex.extend(ttIndex)

X_train = sampleData.loc[trainIndex][['LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
y_train = sampleData.loc[trainIndex]['BAD']
print('Observations(TRAIN PARTITION): ',len(X_train))
print(y_train.value_counts())
print('Mean of DEBTINC(TRAIN PARTITION): ', X_train['DEBTINC'].mean())
print('Standard deviation of DEBTINC(TRAIN PARTITION): ', X_train['DEBTINC'].std())
print('Mean of LOAN(TRAIN PARTITION): ', X_train['LOAN'].mean())
print('Standard deviation of LOAN(TRAIN PARTITION): ', X_train['LOAN'].std())
print('Mean of MORTDUE(TRAIN PARTITION): ', X_train['MORTDUE'].mean())
print('Standard deviation of MORTDUE(TRAIN PARTITION): ', X_train['MORTDUE'].std())
print('Mean of VALUE(TRAIN PARTITION): ', X_train['VALUE'].mean())
print('Standard deviation of VALUE(TRAIN PARTITION): ', X_train['VALUE'].std())

X_test = sampleData.loc[testIndex][['LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
y_test = sampleData.loc[testIndex]['BAD']
print('Observations(TEST PARTITION): ',len(X_test))
print(y_test.value_counts())
print('Mean of DEBTINC(TEST PARTITION): ', X_test['DEBTINC'].mean())
print('Standard deviation of DEBTINC(TEST PARTITION): ', X_test['DEBTINC'].std())
print('Mean of LOAN(TEST PARTITION): ', X_test['LOAN'].mean())
print('Standard deviation of LOAN(TEST PARTITION): ', X_test['LOAN'].std())
print('Mean of MORTDUE(TEST PARTITION): ', X_test['MORTDUE'].mean())
print('Standard deviation of MORTDUE(TEST PARTITION): ', X_test['MORTDUE'].std())
print('Mean of VALUE(TEST PARTITION): ', X_test['VALUE'].mean())
print('Standard deviation of VALUE(TEST PARTITION): ', X_test['VALUE'].std())

# c. We next try the stratified random sampling method.  We use BAD and REASON to jointly define the strata. Since the strata variables may contain missing values, we will replace the missing values in BAD with the integer 99 and in REASON with the string  ‘MISSING’.  What are the frequency distributions of BAD (including missing) in each partition?  What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE in each partition?
sampleData_na = hmeq[['BAD','LOAN','MORTDUE','VALUE','REASON','JOB','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
sampleData_na['BAD'] = sampleData_na['BAD'].fillna(99)
sampleData_na['REASON'] = sampleData_na['REASON'].fillna('MISSING')

print(sampleData_na['BAD'].value_counts())
print(sampleData_na['REASON'].value_counts())

df_train, df_test = train_test_split(sampleData_na, test_size=0.3, random_state=20230101, stratify=sampleData_na[["BAD", "REASON"]])
print('Observations(TRAIN PARTITION): ',len(df_train))
print(df_train['BAD'].value_counts())
print('Mean of DEBTINC(TRAIN PARTITION): ', df_train['DEBTINC'].mean())
print('Standard deviation of DEBTINC(TRAIN PARTITION): ', df_train['DEBTINC'].std())
print('Mean of LOAN(TRAIN PARTITION): ', df_train['LOAN'].mean())
print('Standard deviation of LOAN(TRAIN PARTITION): ', df_train['LOAN'].std())
print('Mean of MORTDUE(TRAIN PARTITION): ', df_train['MORTDUE'].mean())
print('Standard deviation of MORTDUE(TRAIN PARTITION): ', df_train['MORTDUE'].std())
print('Mean of VALUE(TRAIN PARTITION): ', df_train['VALUE'].mean())
print('Standard deviation of VALUE(TRAIN PARTITION): ', df_train['VALUE'].std())

print('Observations(TEST PARTITION): ',len(df_test))
print(df_test['BAD'].value_counts())
print('Mean of DEBTINC(TEST PARTITION): ', df_test['DEBTINC'].mean())
print('Standard deviation of DEBTINC(TEST PARTITION): ', df_test['DEBTINC'].std())
print('Mean of LOAN(TEST PARTITION): ', df_test['LOAN'].mean())
print('Standard deviation of LOAN(TEST PARTITION): ', df_test['LOAN'].std())
print('Mean of MORTDUE(TEST PARTITION): ', df_test['MORTDUE'].mean())
print('Standard deviation of MORTDUE(TEST PARTITION): ', df_test['MORTDUE'].std())
print('Mean of VALUE(TEST PARTITION): ', df_test['VALUE'].mean())
print('Standard deviation of VALUE(TEST PARTITION): ', df_test['VALUE'].std())