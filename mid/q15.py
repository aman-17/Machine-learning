import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neighbors as kNN
import sklearn.svm as svm
import sklearn.tree as tree
import statsmodels.api as sm

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)  
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
# pandas.set_option('precision', 13)
# numpy.set_printoptions(precision = 13)

trainData = pandas.read_csv('./mid/Q15.csv',
                              usecols = ['x', 'y', 'group'])

y_threshold = trainData['group'].mean()

# Build Support Vector Machine classifier

# Convert to the polar coordinates
trainData['radius'] = numpy.sqrt(trainData['x']**2 + trainData['y']**2)
trainData['theta'] = numpy.arctan2(trainData['y'],trainData['x'])                                           

# Build Support Vector Machine classifier
xTrain = trainData[['radius','theta']]
yTrain = trainData['group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20191106, max_iter = -1, decision_function_shape='ovr')
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
w = thisFit.coef_[0]
bSlope = -w[0] / w[1]
xx = numpy.linspace(-3, 3)
aIntercept = (thisFit.intercept_[0]) / w[1]
yy = aIntercept + bSlope * xx

# plot the parallels to the separating hyperplane that pass through the
# support vectors
supV = thisFit.support_vectors_
wVect = thisFit.coef_
crit = thisFit.intercept_[0] + numpy.dot(supV, numpy.transpose(thisFit.coef_))

b = thisFit.support_vectors_[0]
yy_down = (b[1] - bSlope * b[0]) + bSlope * xx
print("Down",b)
b = thisFit.support_vectors_[-1]
yy_up = (b[1] - bSlope * b[0]) + bSlope * xx
print("Up",b)
cc = thisFit.support_vectors_


print('\n\nSupport Vectors',cc)

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green']
plt.figure(dpi=100)
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'x', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
plt.scatter(cc[:,0], cc[:,1], color = 'black', marker = '+', s = 100)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

