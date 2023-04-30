from sklearn.metrics import f1_score
import numpy as np

y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_prob = [0.2, 0.6, 0.4, 0.4, 0.5, 0.5, 0.8, 0.6, 0.7, 0.7]

thresholds = sorted(set(y_prob))
best_threshold = None
best_f1 = -1

for threshold in thresholds:
    y_pred = [int(p >= threshold) for p in y_prob]
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
