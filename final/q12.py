import numpy as np

def entropy(p):
    return -np.sum(p * np.log2(p), where=p>0)

observed = np.array([[1731, 846], [1246, 490], [1412, 543], [2700, 690]])
total = observed.sum()
p = observed.sum(axis=0) / total
root_entropy = entropy(p)

print(f"Root Entropy: {root_entropy}")