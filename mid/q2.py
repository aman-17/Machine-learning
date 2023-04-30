import numpy as np

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return 1 - (dot_product / (norm_a * norm_b))

print(cos_sim([3, 4], [-3, -4]))
print(cos_sim([3, 4], [-1.5, 2]))
print(cos_sim([3, 4], [-1, 0.75]))
print(cos_sim([3, 4], [2, -1.5]))
print(cos_sim([3, 4], [8, 6]))
