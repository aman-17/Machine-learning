import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
train_data = pd.read_csv('./AS5/WineQuality_Train.csv')
test_data = pd.read_csv('./AS5/WineQuality_Test.csv')
X_train = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = train_data['quality_grp']
X_test = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = test_data['quality_grp']
w_train = np.array([1 for i in range(len(X_train))])
y_ens_predProb = np.zeros((len(y_train), 2))
allCombResult=[]
ens_accuracy = 0.0
for itnum in range(1,101):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
    ada_model = AdaBoostClassifier(base_estimator=classTree, n_estimators=itnum)
    treeFit = ada_model.fit(X_train, y_train, w_train)
    y_predProb = ada_model.predict_proba(X_train)
    y_predClass = np.where(y_predProb[:,1] >= 0.5, 1, 0)
    accuracy = np.sum(np.where(y_train == y_predClass, w_train, 0.0)) / np.sum(w_train)
    ens_accuracy = ens_accuracy + accuracy
    y_ens_predProb = y_ens_predProb + accuracy * y_predProb
    if ((1.0 - accuracy) < 0.0000001):
        break
    print('\n')
    print('Iteration = ', itnum)
    print('Weighted Accuracy = ', accuracy)
    print('Weight:\n', w_train)
    print('Predicted Class:\n', y_predClass)
    eventError = np.where(y_train == 1, (1 - y_predProb[:,1]), (0 - y_predProb[:,1]))
    w_train = np.abs(eventError)
    w_train = np.where(y_predClass != y_train, 1.0, 0.0) + w_train
    print('Event Error:\n', eventError)
    y_test_pred = ada_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    allCombResult.append([itnum, accuracy, np.sum(w_train), y_predClass, test_accuracy])
allCombResult = pd.DataFrame(allCombResult, columns = ['Iteration', 'Weighted_Accuracy', 'Weight', 'Predicted_Class', 'Test Accuracy'])
allCombResult.to_csv('as5_q2_new-2.csv')
y_ens_predProb = y_ens_predProb / ens_accuracy
y_ens_predClass = np.where(y_ens_predProb[:,1] >= 0.2, 1, 0)
y_test_proba = ada_model.predict_proba(X_test)
y_test_pred = np.where(y_test_proba[:,1] > 0.2, 1, 0)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = metrics.roc_auc_score(y_test, y_test_proba[:,1])
print("Test AUC score is ", test_auc)
print('Testing Accuracy:', test_accuracy)
test_data['y_predProb'] = y_test_proba[:, 1]
test_data.boxplot(column='y_predProb', by='quality_grp')
plt.title('Predicted Probability for quality_grp = 1')
plt.suptitle("")
plt.show