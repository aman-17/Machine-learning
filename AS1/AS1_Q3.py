import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
fraud = pd.read_csv('./AS1/Fraud.csv', delimiter=',', usecols=['FRAUD','TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS'])

# a. What percent of investigations are found to be frauds?  This is the empirical fraud rate.  Please round your answers to the fourth decimal place.
print(fraud.groupby('FRAUD').size())
a1 = round(np.mean(fraud['FRAUD'])*100,4)
print('Fraud percentage is: ', a1)


# b. We will divide the complete observations into 80% Training and 20% Testing partitions.  A complete observation does not contain missing values in any of the variables.  The random seed is 20230225.  The stratum variable is FRAUD.  How many observations are in each partition?
def simpleRandomSample (obsIndex, trainFraction = 0.8):
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

y = fraud['FRAUD']
sampleData = fraud[['FRAUD','TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']].dropna().reset_index()
random.seed(a = 20230225)
trainIndex = []
testIndex = []

for cat in [0,1]:
   obsIndex = sampleData[sampleData['FRAUD'] == cat].index
   trIndex, ttIndex = simpleRandomSample (obsIndex, trainFraction = 0.8)
   trainIndex.extend(trIndex)
   testIndex.extend(ttIndex)

# df_train, df_test = train_test_split(sampleData, test_size=0.2, stratify=sampleData['FRAUD'])

X_train = sampleData.loc[trainIndex][['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
y_train = sampleData.loc[trainIndex]['FRAUD']
print('Observations in train partition: ',len(X_train))
X_test = sampleData.loc[testIndex][['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
y_test = sampleData.loc[testIndex]['FRAUD']
print('Observations in test partition: ',len(X_test))


# c. Use the KNeighborsClassifier module to train the Nearest Neighbors algorithm.  We will try the number of neighbors from 2 to 7 inclusively. We will classify an observation as a fraud if the proportion of FRAUD = 1 among its neighbors is greater than or equal to the empirical fraud rate (rounded to the fourth decimal place).  What are the misclassification rates of these numbers of neighbors in each partition?
result = []
for k in range(2,8):
    neigh = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', metric = 'euclidean')
    nbrs = neigh.fit(X_train, y_train)
    distances, indices = nbrs.kneighbors(X_train)
    cprob_train = nbrs.predict_proba(X_train)
    score_result_train = nbrs.score(X_train, y_train)
    mce_train, rase_train = nbrs_metric (cprob_train, y_train, 2)
    predict_class_train = np.argmax(cprob_train, axis = 1)
    misclass_rate_train = np.mean(np.where(y_train == predict_class_train, 0, 1))

    cprob_test = nbrs.predict_proba(X_test)
    mce_test, rase_test = nbrs_metric (cprob_test, y_test, 2)
    score_result_test = nbrs.score(X_test, y_test)
    mce_train, rase_train = nbrs_metric (cprob_test, y_test, 2)
    predict_class_test = np.argmax(cprob_test, axis = 1)
    misclass_rate_test = np.mean(np.where(y_test == predict_class_test, 0, 1))
    # print(f'Misclassification Rate in test partition for n = {k}: ', round(misclass_rate_test*100, 4))
    result.append([k, mce_train, mce_test, rase_train, rase_test, round(score_result_train*100, 4), round(misclass_rate_train*100, 4), round(score_result_test*100, 4), round(misclass_rate_test*100, 4)])

result_df = pd.DataFrame(result, columns = ['k', 'MCE_Train', 'MCE_Test', 'RASE_Train', 'RASE_Test', 'Accuracy_Train', 'Misclassification_Train', 'Accuracy_Test', 'Misclassification_Test'])
print(result_df)

result_df.to_csv("q3.csv", index=True)

# d. Which number of neighbors will yield the lowest misclassification rate in the Testing partition?  In the case of ties, choose the smallest number of neighbors. 
min_miss = result_df['Misclassification_Test'].idxmin()
print(f"Lowest Misclassification rate is {result_df['Misclassification_Test'].min()} for k = {result_df['k'][min_miss]}.")

# e. Consider this focal observation where DOCTOR_VISITS is 8, MEMBER_DURATION is 178, NUM_CLAIMS is 0, NUM_MEMBERS is 2, OPTOM_PRESC is 1, and TOTAL_SPEND is 16300.  Use your selected model from Part (d) and find its neighbors.  What are the neighbors’ observation values?  Also, calculate the predicted probability that this observation is a fraud.
sampleData_new = fraud[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
y = fraud['FRAUD']
x = np.matrix(sampleData_new.values)
evals, evecs = np.linalg.eigh(x.transpose() * x)
dvals = 1.0 / np.sqrt(evals)
transf = evecs * np.diagflat(dvals)
transf_x = np.matmul(x, transf)
trainData = pd.DataFrame(transf_x)

kNNSpec = KNeighborsClassifier(n_neighbors=result_df['k'][min_miss], algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData, y)
distances, indices = nbrs.kneighbors(trainData)

class_prob = nbrs.predict_proba(trainData)
score_result = nbrs.score(trainData, y)
print("Score Function: ", score_result)
predict_class = np.argmax(class_prob, axis = 1)
misclass_rate = np.mean(np.where(y == predict_class, 0, 1))
print('Misclassification Rate = ', misclass_rate)

test_df = pd.DataFrame.from_dict({
    'TOTAL_SPEND': [16300],
    'DOCTOR_VISITS': [8],
    'NUM_CLAIMS': [0],
    'MEMBER_DURATION':[178], 
    'OPTOM_PRESC':[1],
    'NUM_MEMBERS':[2]
})

predict = nbrs.predict(test_df)
print("Prediction of FRAUD: ", predict)

score_result = nbrs.score(test_df, [0])
print("Accuracy of prediction: ", score_result*100)

transf_x = np.matmul(test_df, transf)
scoreData = pd.DataFrame(transf_x)
transf_xf = np.matmul(test_df, transf)
dist_f, index_f = nbrs.kneighbors(pd.DataFrame(transf_xf))
for i in index_f:
    print("Neighbor Value: \n", x[i])
    print("Index and FRAUD: \n", y.iloc[i])

print(f"The FRAUD values of all {result_df['k'][min_miss]} neighbors are 0, the predicted probability of fraud of the focal observation is also 0 and, therefore, the predicted FRAUD is 0. The predicted probability of fraud is 0.")