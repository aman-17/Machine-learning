import numpy as np

# Define the frequency distribution of the categorical feature JOB
job_freq = np.array([2388, 1276, 948, 767, 193, 109])

# Calculate the total number of observations
total = np.sum(job_freq)

# Calculate the proportion of each class
job_prop = job_freq / total

# Calculate the Gini Index for the root node
gini_index = 1 - np.sum(job_prop ** 2)

print(f'Gini Index: {gini_index:.4f}')