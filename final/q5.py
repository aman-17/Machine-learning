import pandas as pd
import statsmodels.api as stats

df = pd.read_csv('./final/FinalQ5.csv')
TransportMode = pd.get_dummies(df['TransportMode'].astype('category'))
Y = df['Late4Work']
X = df[['CommuteMile']].join(TransportMode)

logit = stats.MNLogit(Y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
thisLLK = logit.loglike(thisParameter.values)