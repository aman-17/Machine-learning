import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt



# reference: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
def datapreprocess()->pd.DataFrame:
    dataset = pd.read_csv('./AS2/Groceries.csv', delimiter = ',', usecols = ['Customer', 'Item'])
    dataset_list = dataset.groupby(['Customer'])['Item'].apply(list).values.tolist()
    encoding = TransactionEncoder()
    encoding_transform = encoding.fit(dataset_list).transform(dataset_list)
    transform_dataset = pd.DataFrame(encoding_transform, columns=encoding.columns_)
    len_cust = np.zeros(len(dataset_list))
    for items in range(len(dataset_list)):
        len_cust[items] = len(dataset_list[items])
    return dataset, transform_dataset, len(dataset_list)

def assr_rules(max_length, min_support = 0.01, metric = "confidence", min_threshold= 0.01)->float:
    itemsets_support = apriori(datapreprocess()[1], min_support=min_support, use_colnames=True, max_len = max_length)
    assrItemsets = association_rules(itemsets_support, metric = metric, min_threshold = min_threshold)
    return itemsets_support, assrItemsets

# a. What is the number of items in the Universal Set?  What is the maximum number of itemsets that we can find in theory from the data?  What is the maximum number of association rules that we can generate in theory from the data?
noItems = datapreprocess()[0].Item.unique()
print(len(noItems))
a_assrRules = assr_rules(max_length = 32, min_support = 0.01)
print(len(a_assrRules[1]))

# b. We are interested in the itemsets that can be found in the market baskets of at least seventy-five (75) customers.  How many itemsets did we find?  Also, what is the largest number of items, i.e., , among these itemsets?
frequent_itemsets = apriori(datapreprocess()[1], min_support = (75/datapreprocess()[2]), max_len = 32, use_colnames = True)
# print(frequent_itemsets)
# c. We will use up to the largest  value we found in Part (b) and then generate the association rules whose Confidence metrics are greater than or equal to 1%.  How many association rules can we find?  Next, we plot the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for these association rules.  We will use the Lift metrics to indicate the size of the marker.  We will add a color gradient legend to the chart for the Lift metrics.
assrItemsets_new = assr_rules(max_length = 32, min_support = (75/datapreprocess()[2]))[1]

plt.figure(dpi = 100)
plt.scatter(x = assrItemsets_new['confidence'], y = assrItemsets_new['support'], c = assrItemsets_new['lift'], s = assrItemsets_new['lift']*3, cmap='viridis')
plt.grid(True, axis = 'both')
plt.xlim(0.0, 0.7)
plt.ylim(0.0, 0.08)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.colorbar().set_label('Lift')
plt.set_cmap('plasma')
plt.show()


# d. Among the rules that you found in Part (c), list the rules whose Confidence metrics are greater than or equal to 60%.  Please show the rules in a table that shows the Antecedent, the Consequent, the Support, the Confidence, the Expected Confidence, and the Lift.
confidence_sixty = assrItemsets_new[assrItemsets_new['confidence'] >= 0.6]
print(confidence_sixty)
