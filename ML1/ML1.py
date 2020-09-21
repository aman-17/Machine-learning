import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns #FOR GRAPHS

csv=pd.read_csv('d1.csv')
print(csv)

a=[]
for i in range(1000):
    a.append(csv["math score"][i]+csv["reading score"][i]+csv["writing score"][i])

df1=pd.DataFrame(data=a,columns=['total score'])
dataframe=pd.concat([csv,df1],axis=1)
dataframe.drop(["math score"],axis=1,inplace=True)
dataframe.drop(['reading score'],axis=1,inplace=True)
dataframe.drop(['writing score'],axis=1,inplace=True)
print(dataframe)
xc=dataframe.isnull().sum()##HOW MANY VALUES ARE NULL
print(xc)



plt.figure(figsize=(20,10))
xc2=sns.boxplot(
    data=dataframe,
    x='parental level of education',
    y='total score',
    color='red')

print(xc2)
