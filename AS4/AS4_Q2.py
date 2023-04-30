import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import warnings
warnings.filterwarnings("ignore")

import Utility
Y = numpy.array(['Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Non-Event','Event','Event','Event','Event','Event','Event','Event','Event','Event','Event'])

predProbEvent = numpy.array([0.0814,0.1197,0.1969,0.3505,0.3878,0.3940,0.4828,0.4889,0.5587,0.6175,0.4974,0.6732,0.6744,0.6836,0.7475,0.7828,0.6342,0.6527,0.6668,0.5614])

# Calculate the binary model metrics
outSeries = Utility.binary_model_metric (Y, 'Event', 'Non-Event', predProbEvent, eventProbThreshold = 0.5)

print('                  Accuracy: {:.13f}' .format(1.0-outSeries['MCE']))
print('    Misclassification Rate: {:.13f}' .format(outSeries['MCE']))
print('          Area Under Curve: {:.13f}' .format(outSeries['AUC']))
print('Root Average Squared Error: {:.13f}' .format(outSeries['RASE']))

# Generate the coordinates for the ROC curve
outCurve = Utility.curve_coordinates (Y, 'Event', 'Non-Event', predProbEvent)

Threshold = outCurve['Threshold']
Sensitivity = outCurve['Sensitivity']
OneMinusSpecificity = outCurve['OneMinusSpecificity']
Precision = outCurve['Precision']
Recall = outCurve['Recall']
F1Score = outCurve['F1Score']

# Draw the ROC curve
plt.figure(dpi = 80)
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'brown', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'green', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.xticks(numpy.arange(0.0,1.1,0.1))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.show()

# Draw the Kolmogorov Smirnov curve
plt.figure(dpi = 80)
plt.plot(Threshold, Sensitivity, marker = 'o', label = 'True Positive',
         color = 'brown', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(Threshold, OneMinusSpecificity, marker = 'o', label = 'False Positive',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

# Draw the Precision-Recall curve
plt.figure(dpi = 80)
plt.plot(Recall, Precision, marker = 'o',
         color = 'brown', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [6/11, 6/11], color = 'green', linestyle = ':', label = 'No Skill')
plt.grid(True)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Draw the F1 Score curve
plt.figure(dpi = 80)
plt.plot(Threshold, F1Score, marker = 'o',
         color = 'black', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.show()

from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

event_probs = [0.4974, 0.6732, 0.6744, 0.6836, 0.7475, 0.7828, 0.6342, 0.6527, 0.6668, 0.5614]
nonevent_probs = [0.0814, 0.1197, 0.1969, 0.3505, 0.3878, 0.3940, 0.4828, 0.4889, 0.5587, 0.6175]

y_true = [1] * len(event_probs) + [0] * len(nonevent_probs)
y_scores = event_probs + nonevent_probs

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
thresholds = np.append(thresholds, 1)

misclassification_rates = []
for threshold in thresholds:
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    misclassification_rate = (fp + fn) / (tp + tn + fp + fn)
    misclassification_rates.append(misclassification_rate)

# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()

f1_scores = 2 * recall * precision / (recall + precision)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
best_f1_score = f1_scores[best_threshold_index]
best_misclassification_rate = misclassification_rates[best_threshold_index]

print(f'The probability threshold that yields the highest F1 Score is {best_threshold:.4f} with an F1 Score of {best_f1_score:.4f} and a misclassification rate of {best_misclassification_rate:.4f}.')

from sklearn.metrics import roc_curve, confusion_matrix

y_true = [1] * len(event_probs) + [0] * len(nonevent_probs)
y_scores = event_probs + nonevent_probs

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
ks_statistic = np.max(tpr - fpr)
best_threshold_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_threshold_index]

y_pred = [1 if score >= best_threshold else 0 for score in y_scores]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
misclassification_rate = (fp + fn) / (tp + tn + fp + fn)

print(f'The probability threshold that yields the highest Kolmogorov-Smirnov statistic is {best_threshold:.4f} with a Kolmogorov-Smirnov statistic of {ks_statistic:.4f} and a misclassification rate of {misclassification_rate:.4f}.')
