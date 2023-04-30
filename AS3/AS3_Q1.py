import numpy as np 
import pandas as pd
import math
from sklearn import metrics
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import preprocessing, naive_bayes
import itertools

import warnings
warnings.filterwarnings("ignore")

#reference : https://www.kaggle.com/code/opawar600/decision-tree-from-scratch

df = pd.read_excel("./AS3/claim_history.xlsx")
features = df[["CAR_TYPE","OCCUPATION","EDUCATION"]]
labels = df['CAR_USE']
features["Labels"] = labels

cross_Table_Train = pd.crosstab(labels,columns =  ["Count"],margins=True,dropna=True)
cross_Table_Train["Proportions"] = (cross_Table_Train["Count"]/len(labels))*100
print(cross_Table_Train)


#All possible combinations for occupation
occupation_column = df["OCCUPATION"].unique()
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))


car_type_column = df["CAR_TYPE"].unique()
car_type_combinations = []
for i in range(1,math.ceil(len(car_type_column)/2)+1):
    x = list(combinations(car_type_column,i))
    if i == 3:
        x = x[:10]
    car_type_combinations.extend(x) 

education_combinations = [("Below High Sc",),("Below High Sc","High School",),("Below High Sc","High School","Bachelors",),("Below High Sc","High School","Bachelors","Masters",),("High School', 'Bachelors', 'Masters', 'PhD",)]

def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   print('Split: ',split)
   dataTable = inData
   dataTable['LE_Split'] = False
   for k in dataTable.index:
       if dataTable.iloc[:,0][k] in split:
           dataTable['LE_Split'][k] = True
#    print(dataTable['LE_Split'])
   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
#    print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * np.log2(proportion)
      print('Row = ', iRow, 'Entropy =', rowEntropy)
      
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
      
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
   print(tableEntropy)
   print(' ')
  
   return(tableEntropy)

def calculate_min_entropy(df,variable,combinations):
    inData1 = df[[variable,"Labels"]]
    entropies = []
    for i in combinations:
        EV = EntropyIntervalSplit(inData1, list(i))
        entropies.append((EV,i))
    return min(entropies)

entropy_occupation = calculate_min_entropy(features,"OCCUPATION",occupation_combinations)
print(entropy_occupation)

entropy_cartype = calculate_min_entropy(features,"CAR_TYPE",car_type_combinations)
print(entropy_cartype)

entropy_education = calculate_min_entropy(features,"EDUCATION",education_combinations)
print(entropy_education)


df_1_left = features[(features["OCCUPATION"] == "Blue Collar") | (features["OCCUPATION"] == "Unknown") | (features["OCCUPATION"] == "Student")]
df_1_right =  features[(features["OCCUPATION"] != "Blue Collar") & (features["OCCUPATION"] != "Unknown") & (features["OCCUPATION"] != "Student")]

left_edu_entropy = calculate_min_entropy(df_1_left,"EDUCATION",education_combinations)
print(left_edu_entropy)

left_ct_entropy = calculate_min_entropy(df_1_left,"CAR_TYPE",car_type_combinations)
print(left_ct_entropy)

occupation_column = ['Blue Collar', 'Unknown', 'Student']
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))
left_occupation_entropy = calculate_min_entropy(df_1_left,"OCCUPATION",occupation_combinations)
print(occupation_combinations)

occupation_column = ['Professional', 'Manager', 'Clerical', 'Doctor','Lawyer','Home Maker']
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))

right_edu_entropy = calculate_min_entropy(df_1_right,"EDUCATION",education_combinations)
right_ct_entropy = calculate_min_entropy(df_1_right,"CAR_TYPE",car_type_combinations)
print(right_ct_entropy , right_edu_entropy)

education_combinations = [("High School', 'Bachelors', 'Masters', 'PhD",)]


df_2_left_left = df_1_left[(features["EDUCATION"] == "Below High Sc")]
df_2_left_right = df_1_left[(features["EDUCATION"] != "Below High Sc")]

df_2_right_left = df_1_right[(features["CAR_TYPE"] == "Minivan") | (features["CAR_TYPE"] == "Sports Car") | (features["CAR_TYPE"] == "SUV")]
df_2_right_right = df_1_right[(features["CAR_TYPE"] != "Minivan") & (features["CAR_TYPE"] != "Sports Car") & (features["CAR_TYPE"] != "SUV")]

right_ct_entropy = calculate_min_entropy(df_2_left_left,"EDUCATION",education_combinations)
print(right_ct_entropy)

occupation_column = ["Blue Collar","Student","Unknown"]
occupation_combinations = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occupation_combinations+=list(combinations(occupation_column,i))
entropy_occupation = calculate_min_entropy(df_2_left_left,"EDUCATION",education_combinations)
print(entropy_occupation)

cnt = 0
for i in features["Labels"]:
    if i == "Commercial":
        cnt+=1
threshold = cnt/len(features["Labels"])
print("Threshold probability of an Commercial is given as",threshold)


cnt = 0
for i in features["Labels"]:
    if i == "Private":
        cnt+=1
threshold = cnt/len(features["Labels"])
print("Threshold probability of an Private is given as",threshold)


predicted_probability=[]
occ = ["Blue Collar","Student","Unknown"]
edu = ["Below High School",]
cartype = ["Minivan","SUV","Sports Car"]
for k in features.index:
    if features.iloc[:,1][k] in occ:
            if features.iloc[:,2][k] in edu:
                predicted_probability.append(0.2625)  #Leftmost Leaf Node
            else:
                predicted_probability.append(0.8448)   #Right leaf from left subtree
    else:
            if features.iloc[:,0][k] in cartype:
                predicted_probability.append(0.0065)  #Left leaf from right subtree
            else:
                predicted_probability.append(0.5302)   #Rightmost Leaf Node

prediction = []
for i in range(0,len(labels)):
    if predicted_probability[i] >= threshold :
        prediction.append("Commercial")
    else:
        prediction.append("Private")

p1 = 0.7375
p2 = 0.1552
p3 = 0.9935
p4 = 0.4698

y1 = 607
y2 = 470
y3 = 4564
y4 = 872

p = [p1,p2,p3,p4]
y = [y1,y2,y3,y4]
bin_width = 0.05
bins = np.arange(0, 1 + bin_width, bin_width)

plt.hist(p,weights=y,bins=bins,density=True)
plt.xlabel('Predicted Probability of CAR_USE = Private')
plt.ylabel('Proportions of Observations')
plt.show()

from sklearn.metrics import accuracy_score
print("Missclassification Rate", 1-accuracy_score(labels,prediction))

