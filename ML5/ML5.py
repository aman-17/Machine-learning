import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

names = ['age','sex','chest_pain','blood_pressure','serum cholestoral','Fasting_blood_sugar','ElectroCardioGraphic','Max_heartrate','Induced_angina','St_depression','slope','No_of_vessels','thal','diagnosis']
dataset = pd.read_csv(url, names=names)

data = dataset.drop('thal', axis=1)
print(data.head(20))

array = data.values

X = array[:,1:11]
Y = array[:,12]
Y=Y.astype('int')
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=0)


knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)
predictions = knn.predict(x_test)
print("\n KNN- Classifier\n")
print(accuracy_score(y_test, predictions))


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
predictions = rfc.predict(x_test)
print("\nRandomForest\n")
print(accuracy_score(y_test, predictions))

dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
predictions = dtc.predict(x_test)
print("\nDecision Tree\n")

print(accuracy_score(y_test, predictions))

lr = LogisticRegression()
lr.fit(x_train,y_train)
predictions = lr.predict(x_test)
print("\nLogistic Regression\n")
print(accuracy_score(y_test, predictions))


heart_prediction = lr.predict([[1,3, 172, 199, 1, 0, 162,0, 0.5,1]])
print(heart_prediction)
