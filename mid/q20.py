from sklearn.metrics import silhouette_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")

K = 20
min_score = 1.0  # initialize to a high value
min_N = -1  # initialize to an invalid value
for N in range(K+1, 1001):
    x = np.array(list(range(N))).reshape(-1, 1)
    c = [i % K for i in range(N)]
    score = silhouette_score(x, c)
    if score < min_score:
        min_score = score
        min_N = N

print(f"Lowest Silhouette Score: {min_score:.4f}, N = {min_N}")

# N = 39
# K = 20
# min_score = 1.0 
# min_N = -1

# # Generate data
# x = np.arange(N).reshape(-1, 1)
# c = np.mod(x, K)

# # Compute Silhouette Score
# score = silhouette_score(x, c)
# if score < min_score:
#     min_score = score
#     min_N = N

# print(f"Lowest Silhouette Score: {min_score:.4f}, N = {min_N}")
