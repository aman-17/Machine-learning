import math 
from sklearn.metrics import roc_auc_score

a1 = 22/22
b1 = 2/31 
c1 = 32/34
d1 = 42/42 
# e1 = 1456/2750

a0 = 0/22
b0 = 29/31 
c0 = 2/34 
d0 = 0/42 
# e0 = 1294/2750 

print("Prob of cash at leaf node",a0,b0,c0,d0)
print("Prob of credit at leaf node",a1,b1,c1,d1) 

t = ((1-a1)**2 + (1-b1)**2 + (1-c1)**2 + (1-d1)**2 +  (0-a0)**2 + (0-b0)**2 + (0-c0)**2 + (0-d0)**2 )/8
print("\nRoot Mean Square Error =", "%.4f"% math.sqrt(t)) 

def create_actual_prediction_arrays(n_pos, n_neg):
    prob = n_pos / (n_pos + n_neg)
    y_true = [1] * n_pos + [0] * n_neg
    y_score = [prob] * (n_pos + n_neg)
    
    return y_true, y_score

total_y_true = []
total_y_score = []
for n_pos, n_neg in [(22, 0), (2, 29), (32, 2), (42, 0)]:
    y_true, y_score = create_actual_prediction_arrays(n_pos, n_neg)
    total_y_true += y_true
    total_y_score += y_score
    
print("auc_score = ", roc_auc_score(y_true=total_y_true, y_score=total_y_score))
