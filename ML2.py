import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.DESCR)
cancer.keys()
def answer_zero():
    return len(cancer['feature_names'])
answer_zero()

def answer_one():
    return pd.DataFrame(np.c_[cancer['data'], cancer['target']],
columns= np.append(cancer['feature_names'], ['target']))

answer_one()

def answer_two():
    cancerdf = answer_one()
    malignant = len(cancerdf[cancerdf['target'] == 0])
    benign = len(cancerdf[cancerdf['target'] == 1])
    target = pd.Series(data = [malignant, benign], index = ['malignant', 'benign'])
    return target
answer_two()

def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:, :30]
    y = cancerdf.iloc[:, 30]
    return X, y
answer_three()

from sklearn.model_selection import train_test_split
def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
answer_four()

from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()

    knn = KNeighborsClassifier(n_neighbors = 1)
    return knn.fit(X_train, y_train)
answer_five()

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)

    predict = answer_five()
    label = predict.predict(means)
    return label
print(answer_six())

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()

    return knn.predict(X_test)
print(answer_seven())

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    
    
    return knn.score(X_test, y_test)

print(answer_eight())

