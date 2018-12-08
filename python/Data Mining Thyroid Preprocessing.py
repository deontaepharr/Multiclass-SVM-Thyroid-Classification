#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data
# ----

# In[ ]:


columns = ["Age", "Sex", "On Thyroxine", "Query on Thyroxine", 
           "On Antithyroid Medication", "Sick", "Pregnant", 
           "Thyroid Surgery", "I131 Treatment", "Query Hypothyroid", 
           "Query Hyperthyroid", "Lithium", "Goitre", "Tumor", 
           "Hypopituitary", " Psych", "TSH Measured", "TSH", "T3 Measured", 
           "T3", "TT4 Measured", "TT4", "T4U Measured", "T4U", 
           "FTI Measured", "FTI", "TBG Measured", "TBG", "Referral Source", "Category"]


# In[ ]:


hyper_data = pd.read_csv("allhyper.csv", names=columns)
hypo_data = pd.read_csv("allhypo.csv", names=columns)
sick_data = pd.read_csv("sick.csv", names=columns)


# In[ ]:


def parse_row(row):
    row = row.split(".")[0]
    return row


# # Hyperthyroid Data
# ---

# In[ ]:


hyper_data['Category'] = hyper_data['Category'].apply(parse_row)


# In[ ]:


def to_hyperthyroid(row):
    if row != "negative":
        row = "hyperthyroid"
    return row


# In[ ]:


hyper_data['Category'] = hyper_data['Category'].apply(to_hyperthyroid)


# In[ ]:


hyper_data['Category'].value_counts()


# In[ ]:


hyper_data.head()


# # Hypothyroid Data
# ---

# In[ ]:


hypo_data['Category'] = hypo_data['Category'].apply(parse_row)


# In[ ]:


def to_hypothyroid(row):
    if row != "negative":
        row = "hypothyroid"
    return row


# In[ ]:


hypo_data['Category'] = hypo_data['Category'].apply(to_hypothyroid)


# In[ ]:


hypo_data['Category'].value_counts()


# In[ ]:


hypo_data.head()


# # Sick Thyroid Data
# ---

# In[ ]:


sick_data['Category'] = sick_data['Category'].apply(parse_row)


# In[ ]:


sick_data['Category'].value_counts()


# In[ ]:


sick_data.head()


# # Combined Thyroid Data
# ---

# In[ ]:


thyroid_frames = [hyper_data, hypo_data, sick_data]


# In[ ]:


thyroid_data = pd.concat(thyroid_frames).drop_duplicates()


# In[ ]:


thyroid_data = thyroid_data.drop('Referral Source', axis=1)


# In[ ]:


thyroid_data["Category"].value_counts()


# In[ ]:


thyroid_data.columns


# # Missing Data Preprocessing
# ----

# In[ ]:


def find_missing_columns(dataframe, print_cols = False):
    # Find columns with too many missing values
    count = 1
    missing_data_columns = []

    for column in dataframe:
        if print_cols:
            print(count, ":", column, "\n", dataframe[column].unique(), "\n", dataframe[column].value_counts())
            print("Missing data? ", "?" in dataframe[column].unique())
            print("\n=========\n")
            count += 1
        
        if "?" in dataframe[column].unique():
            missing_data_columns.append(column)

    return missing_data_columns


# In[ ]:


find_missing_columns(thyroid_data)


# In[ ]:


'''
Drop Columns due to too many missing data:
'TBG'
'TBG Measured'
'''
thyroid_data = thyroid_data.drop('TBG', axis=1)
thyroid_data = thyroid_data.drop("TBG Measured", axis=1)


# In[ ]:


thyroid_data.head()


# # Missing Data Preprocessing 
# - Identify missing and abnormal data

# In[ ]:


def fill_missing_data(dataframe, category_label):
    category = dataframe[category_label]
    cat_missing = category[category != '?']
    cat_val = pd.to_numeric(cat_missing).mean()
    condition = dataframe[category_label]== '?'
    dataframe.loc[condition, category_label] = cat_val


# In[ ]:


def print_column(dataframe, column):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataframe[column])


# ## Age

# In[ ]:


thyroid_data.loc[thyroid_data['Age'] == '455', 'Age'] = '45'
fill_missing_data(thyroid_data, 'Age')
# print_column(thyroid_data, 'Age')


# ## TSH

# In[ ]:


fill_missing_data(thyroid_data, 'TSH')
# print_column(thyroid_data, 'TSH')


# ## T3

# In[ ]:


fill_missing_data(thyroid_data, 'T3')
# print_column(thyroid_data, 'T3')


# ## TT4

# In[ ]:


fill_missing_data(thyroid_data, 'TT4')
# print_column(thyroid_data, 'TT4')


# ## T4U

# In[ ]:


fill_missing_data(thyroid_data, 'T4U')
# print_column(thyroid_data, 'T4U')


# ## FTI

# In[ ]:


fill_missing_data(thyroid_data, 'FTI')
# print_column(thyroid_data, 'FTI')


# ## Sex

# In[ ]:


thyroid_data = thyroid_data[thyroid_data['Sex'] != '?']


# In[ ]:


# Last check to ensure no missing data
find_missing_columns(thyroid_data)


# # Separate Columns for Further Processing
# ------

# In[ ]:


def separate_columns(dataframe):
    continuous = []
    categorical = []
    
    for column in dataframe:
        try:
            pd.to_numeric(dataframe[column]) # a hack, but oh well, it works lmfao
            continuous.append(column)
        except ValueError:
            if column != 'Category' and column != 'Sex':
                categorical.append(column)
            continue

    return continuous,categorical


# In[ ]:


continuous_cols, categorical_cols = separate_columns(thyroid_data)


# # Scale Continuous Data
# -----

# In[ ]:


def normalize_column(dataframe, column):
    dataframe[column] = dataframe[column].apply(pd.to_numeric)
    dataframe[column] = ((dataframe[column]-dataframe[column].min())/
                        (dataframe[column].max()-dataframe[column].min()))


# In[ ]:


for col in continuous_cols:
    normalize_column(thyroid_data, col)


# # Binarize Categorical Data
# -----

# In[ ]:


# Binarize Sex column
conditionF = thyroid_data['Sex'] == 'F'
conditionM = thyroid_data['Sex'] == 'M'

thyroid_data.loc[conditionF, 'Sex'] = 0
thyroid_data.loc[conditionM, 'Sex'] = 1


# In[ ]:


def convert_category(dataframe, column):
    conditionF = dataframe[column] == 'f'
    conditionT = dataframe[column] == 't'

    dataframe.loc[conditionF, column] = 0
    dataframe.loc[conditionT, column] = 1


# In[ ]:


# Binarize T/F columns
for col in categorical_cols:
    convert_category(thyroid_data, col)


# # Balance Categorical Data
# -------

# In[ ]:


def dataset_to_sample(dataset, sample=True):
    if sample:
        without_negative_class = dataset[dataset['Category'] != 'negative']
        negative_class = dataset[dataset['Category'] == 'negative']
        negative_samples = negative_class.sample(220) # Size of the 2nd most data
        return without_negative_class.append(negative_samples)
    else:
        return dataset


# # Save Data To CSV File
# ----------

# ### Balanced Data

# In[ ]:


thyroid_data_balanced = dataset_to_sample(thyroid_data, sample=True)


# In[ ]:


thyroid_data_balanced["Category"].value_counts()


# In[ ]:


thyroid_data_balanced.to_csv("/Users/deontaepharr/Desktop/Projects/DM_Project/thyroid_balanced.csv")


# ### Unbalanced Data

# In[ ]:


thyroid_data_unbalanced = dataset_to_sample(thyroid_data, sample=False)


# In[ ]:


thyroid_data_unbalanced["Category"].value_counts()


# In[ ]:


thyroid_data_unbalanced.to_csv("/Users/deontaepharr/Desktop/Projects/DM_Project/thyroid_unbalanced.csv")

