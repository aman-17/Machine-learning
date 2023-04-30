import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

from scipy.stats import norm
from sklearn import metrics, naive_bayes

# Specify the roles
feature = ['height', 'weight']
target = 'bmi_status'

# Read the Excel file
input_data = pandas.read_excel('./mid/bmi_men.xlsx')
bmi_men = input_data[feature + [target]].dropna().reset_index(drop = True)

cmap = ['blue', 'green', 'orange', 'red']
slabel = ['Below', 'Normal', 'Over', 'Obese']
plt.figure(figsize = (8,6), dpi = 80)
for status in range(4):
    plot_data = bmi_men[bmi_men['bmi_status'] == (status+1)]
    plt.scatter(plot_data['weight'], plot_data['height'], c = cmap[status], label = slabel[status])
plt.xlabel('Weight in Kilograms')
plt.ylabel('Height in Meters')
plt.xticks(numpy.arange(50, 130, 10))
plt.yticks(numpy.arange(1.0, 2.8, 0.2))
plt.grid(axis = 'both', linestyle = '--')
plt.legend(title = 'BMI Level')
# plt.show()

xTrain = bmi_men[feature]
yTrain = bmi_men[target].astype('category')

_objNB = naive_bayes.GaussianNB()
thisFit = _objNB.fit(xTrain, yTrain)

# print('Probability of each target class')
# print(thisFit.class_prior_)

# print('Means of Features of each target class')
# print(thisFit.theta_)

# print('Variances of Features of each target class')
# print(thisFit.var_)

# print('Number of samples encountered for each class during fitting')
# print(thisFit.class_count_)

yTrain_predProb = _objNB.predict_proba(xTrain)

yTrain_predClass = _objNB.predict(xTrain)

confusion_matrix = metrics.confusion_matrix(yTrain, yTrain_predClass)

# Manually calculate the predicted probability
class_prob = bmi_men.groupby(target).size() / bmi_men.shape[0]

summary_height = bmi_men.groupby(target)['height'].describe()
summary_weight = bmi_men.groupby(target)['weight'].describe()

logpdf_height = norm.logpdf(1.85, loc = summary_height['mean'], scale = summary_height['std'])
logpdf_weight = norm.logpdf(96.0, loc = summary_weight['mean'], scale = summary_weight['std'])

logpdf = numpy.log(class_prob) + logpdf_weight + logpdf_height
my_prob = numpy.exp(logpdf)
sum_prob = numpy.sum(my_prob)
my_prob = numpy.divide(my_prob, sum_prob)

xTest = pandas.DataFrame({'height': [1.85], 'weight': [96.0]})
yTest_predProb = _objNB.predict_proba(xTest)

print(my_prob)
print(yTest_predProb)
