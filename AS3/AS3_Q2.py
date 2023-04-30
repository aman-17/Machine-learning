import numpy
import pandas
import sys
import itertools
from sklearn import preprocessing, naive_bayes
import matplotlib.pyplot as plt

df = pandas.read_excel("./AS3/claim_history.xlsx")
features = ["CAR_TYPE","OCCUPATION","EDUCATION"]
labels = 'CAR_USE'
trainData = df[features + [labels]].dropna().reset_index(drop = True)
thresholdT = pandas.crosstab(index = 'Total', columns = trainData[labels], margins = False, dropna = True)
print('Frequency Count:')
print(thresholdT)
thresholdT = thresholdT.div(thresholdT.sum(1), axis = 'index')
table_total = {}
table_total[labels] = thresholdT
print('Row Probability:')
print(table_total[labels])

for i in features:
    print('Feature: ', i)
    thresholdT = pandas.crosstab(index = trainData[i], columns = trainData[labels],
                                margins = False, dropna = True)
    print('Frequency Count:')
    print(thresholdT)
    thresholdT = thresholdT.div(thresholdT.sum(0), axis = 'columns')
    table_total[i] = thresholdT
    print('Row Probability:')
    print(table_total[i])

feature = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']
labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(trainData[labels])
yLabel = labelEnc.inverse_transform([0, 1])
cartype_u = numpy.unique(trainData['CAR_TYPE'])
occ_u = numpy.unique(trainData['OCCUPATION'])
edu_u = numpy.unique(trainData['EDUCATION'])

cat_all = [cartype_u, occ_u, edu_u]
print(cat_all)

featureEnc = preprocessing.OrdinalEncoder(categories = cat_all)
xTrain = featureEnc.fit_transform(trainData[feature])

_objNB = naive_bayes.CategoricalNB(alpha = 0.01)
y_pred = _objNB.fit(xTrain, yTrain).predict(xTrain)
prediction=_objNB.predict_proba(xTrain)[:,1]
print("Number of mislabeled points out of a total %d points : %d"
      % (xTrain.shape[0], (yTrain != y_pred).sum()))

xTest = pandas.DataFrame(list(itertools.product(cartype_u, occ_u, edu_u)), columns = feature)
yTest_predProb = pandas.DataFrame(_objNB.predict_proba(featureEnc.fit_transform(xTest)), columns = ['Commercial', 'Private'])
yTest_score = pandas.concat([xTest, yTest_predProb], axis = 1)

private_mask = yTrain == 1
private_prob = _objNB.predict_proba(xTrain[private_mask])[:, 1]
print(len(xTrain[private_mask]))
bin_width = 0.05
bins = numpy.arange(0, 1 + bin_width, bin_width)

plt.hist(private_prob, bins=bins, density=True)
plt.xlabel('Predicted Probability of CAR_USE = Private')
plt.ylabel('Proportion of Observations')
plt.show()

# # yTest_score.to_csv('naive_bayes.csv', sep=',', encoding='utf-8')
print(yTest_predProb)


