import numpy as np
from scipy.stats import chi2_contingency

observed = np.array([[1731, 846], [1246, 490], [1412, 543], [2700, 690]])
chi2, _, _, expected = chi2_contingency(observed)

n = observed.sum()
r, c = observed.shape
cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))

print(f"Cramer's V: {cramers_v}")
