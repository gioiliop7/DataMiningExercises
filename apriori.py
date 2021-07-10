#GIORGOS ILIOPOULOS 3980 ASKISI 8

import sys
import numpy as np
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

# Loading the Data
data = pd.read_excel('Online_Retail_pract.xlsx')
print(data.head(5))

# Exploring the columns of the data
data.columns
# Exploring the different regions of transactions
print(data.Country.unique())

# Cleanning the data
# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]

# Transactions done in Netherlands
basket_Netherlands = (data[data['Country'] == "Netherlands"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Transactions done in Norway
basket_Norway = (data[data['Country'] == "Norway"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Transactions done in Spain
basket_Spain = (data[data['Country'] == "Spain"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Transactions done in Portugal
basket_Portugal = (data[data['Country'] == "Portugal"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Transactions done in Poland
basket_Poland = (data[data['Country'] == "Poland"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))


# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1


# Encoding the datasets for Netherlands
basket_encoded = basket_Netherlands.applymap(hot_encode)
basket_Netherlands = basket_encoded

# Building the model
print("Holland\n")
frq_items = apriori(basket_Netherlands, min_support=0.05, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(5))


# Encoding the datasets for Norway
basket_encoded = basket_Norway.applymap(hot_encode)
basket_Norway = basket_encoded

print("Norway\n")
# Building the model
frq_items = apriori(basket_Norway, min_support=0.05, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(5))

# Encoding the datasets for Spain
basket_encoded = basket_Spain.applymap(hot_encode)
basket_Spain = basket_encoded

print("Spain\n")
# Building the model
frq_items = apriori(basket_Spain, min_support=0.05, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(5))


# Encoding the datasets for Portugal
basket_encoded = basket_Portugal.applymap(hot_encode)
basket_Portugal = basket_encoded
print("Portugal\n")
# Building the model
frq_items = apriori(basket_Portugal, min_support=0.05, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(5))


# Encoding the datasets for Poland
basket_encoded = basket_Poland.applymap(hot_encode)
basket_Poland = basket_encoded

print("Poland\n")
# Building the model
frq_items = apriori(basket_Poland, min_support=0.05, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(5))