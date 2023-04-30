import numpy as np

def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def weighted_entropy(p1, n1, p2, n2):
    N = n1 + n2
    return n1 / N * entropy(p1 / n1) + n2 / N * entropy(p2 / n2)

n = [1731, 1246, 1412, 2700]
p = [846, 490, 543, 690]

N = sum(n)
P = sum(p)

min_entropy = float('inf')
best_threshold = None

for i in range(3):
    e = weighted_entropy(sum(p[:i+1]), sum(n[:i+1]), sum(p[i+1:]), sum(n[i+1:]))
    if e < min_entropy:
        min_entropy = e
        best_threshold = i

thresholds = ['1 to 3', '4 to 7', '8 to 10', '11+']
print(f"Best separation: {{{', '.join(thresholds[:best_threshold+1])}}} and {{{', '.join(thresholds[best_threshold+1:])}}}")