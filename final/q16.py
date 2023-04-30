import pandas as pd
import numpy as np
import statsmodels.api as stats

train_data = pd.read_csv('./AS5/WineQuality_Train.csv')
test_data = pd.read_csv('./AS5/WineQuality_Test.csv')
threshold = 0.1961733010776
Xtrain = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtrain = stats.add_constant(Xtrain, prepend=True)
Ytrain = train_data['quality_grp']

Xtest = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtest = stats.add_constant(Xtest, prepend=True)
Ytest = test_data['quality_grp']

logit = stats.MNLogit(Ytrain, Xtrain)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

pred_prob = thisFit.predict(Xtest)

predictions = []
count = 0
for i in range(len(pred_prob)):
    if pred_prob[1][i] >= threshold:
        predictions.append(1)
    else:
        predictions.append(0)
        
for k in range(len(Ytest)):
    if Ytest[k] != predictions[k]:
        count += 1
        
MNL_misclassification = count / len(Ytest)
print('The accuracy for the Multinomial Logistic model =', 1-MNL_misclassification)