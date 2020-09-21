#For this assignment, you will be using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer #imports data
cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description

#This function shows how many features does the breast cancer dataset have.
def answer_zero():
    return len(cancer['feature_names'])
answer_zero()

#Converting into DataFrame as it help make many things easier.
def answer_one():
    return pd.DataFrame(np.c_[cancer['data'], cancer['target']],
columns= np.append(cancer['feature_names'], ['target']))
answer_one()

#This function returns a series named target of length 2 with integer values and index = ['malignant', 'benign'] also called as class distribution.
def answer_two():
    cancerdf = answer_one()
    malignant = len(cancerdf[cancerdf['target'] == 0])
    benign = len(cancerdf[cancerdf['target'] == 1])
    target = pd.Series(data = [malignant, benign], index = ['malignant', 'benign'])
    return target
answer_two()

#Split the DataFrame into X (the data) and y (the labels).
def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:, :30]
    y = cancerdf.iloc[:, 30]
    return X, y
answer_three()

#Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
#Set the random number generator state to 0 using random_state=0 to make sure your results match the autograder.
from sklearn.model_selection import train_test_split
def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
answer_four()

#Using KNeighborsClassifier, fitting a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    return knn.fit(X_train, y_train)
answer_five()

#Using your knn classifier, predict the class label using the mean value for each feature. This function should return a numpy array either array([ 0.]) or array([ 1.])
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    predict = answer_five()
    label = predict.predict(means)
    return label
print(answer_six())

#Using your knn classifier, predict the class labels for the test set X_test.
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_test)
print(answer_seven())

#Finding the score (mean accuracy) of your knn classifier using X_test and y_test.
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.score(X_test, y_test)

print(answer_eight())

