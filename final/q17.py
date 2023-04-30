import pandas as pd
import numpy as np
from sklearn.svm import SVC

train_data = pd.read_csv('./AS5/WineQuality_Train.csv')
test_data = pd.read_csv('./AS5/WineQuality_Test.csv')
threshold = 0.1961733010776

Xtrain = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Ytrain = train_data['quality_grp']
Xtest = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Ytest = test_data['quality_grp']

svm_model = SVC(kernel='linear', random_state=2023484, max_iter=-1)
thisFit = svm_model.fit(Xtrain,Ytrain)
pred_prob = thisFit.predict(Xtest)

predictions = []
count = 0
for i in range(len(pred_prob)):
    if pred_prob[i] >= threshold:
        predictions.append(1)
    else:
        predictions.append(0)
for k in range(len(Ytest)):
    if Ytest[k] != predictions[k]:
        count += 1

SVM_misclassification = count / len(Ytest)
print('The accuracy for the SVM model =', 1-SVM_misclassification)
