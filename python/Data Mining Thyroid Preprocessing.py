import pandas as pd


# # Load Data
# ----

columns = ["Age", "Sex", "On Thyroxine", "Query on Thyroxine", 
           "On Antithyroid Medication", "Sick", "Pregnant", 
           "Thyroid Surgery", "I131 Treatment", "Query Hypothyroid", 
           "Query Hyperthyroid", "Lithium", "Goitre", "Tumor", 
           "Hypopituitary", " Psych", "TSH Measured", "TSH", "T3 Measured", 
           "T3", "TT4 Measured", "TT4", "T4U Measured", "T4U", 
           "FTI Measured", "FTI", "TBG Measured", "TBG", "Referral Source", "Category"]


hyper_data = pd.read_csv("/Users/deontaepharr/Documents/GitHub/multiclass_svm_thyroid_classification/datasets/allhyper.csv", names=columns)
hypo_data = pd.read_csv("/Users/deontaepharr/Documents/GitHub/multiclass_svm_thyroid_classification/datasets/allhypo.csv", names=columns)
sick_data = pd.read_csv("/Users/deontaepharr/Documents/GitHub/multiclass_svm_thyroid_classification/datasets/sick.csv", names=columns)


def parse_row(row):
    row = row.split(".")[0]
    return row


# # Hyperthyroid Data
# ---


hyper_data['Category'] = hyper_data['Category'].apply(parse_row)


def to_hyperthyroid(row):
    if row != "negative":
        row = "hyperthyroid"
    return row


hyper_data['Category'] = hyper_data['Category'].apply(to_hyperthyroid)

hyper_data['Category'].value_counts()

hyper_data.head()


# # Hypothyroid Data
# ---




hypo_data['Category'] = hypo_data['Category'].apply(parse_row)





def to_hypothyroid(row):
    if row != "negative":
        row = "hypothyroid"
    return row





hypo_data['Category'] = hypo_data['Category'].apply(to_hypothyroid)





hypo_data['Category'].value_counts()





hypo_data.head()


# # Sick Thyroid Data
# ---




sick_data['Category'] = sick_data['Category'].apply(parse_row)





sick_data['Category'].value_counts()





sick_data.head()


# # Combined Thyroid Data
# ---




thyroid_frames = [hyper_data, hypo_data, sick_data]





thyroid_data = pd.concat(thyroid_frames).drop_duplicates()





thyroid_data = thyroid_data.drop('Referral Source', axis=1)





thyroid_data["Category"].value_counts()





thyroid_data.columns


# # Missing Data Preprocessing
# ----




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





find_missing_columns(thyroid_data)





'''
Drop Columns due to too many missing data:
'TBG'
'TBG Measured'
'''
thyroid_data = thyroid_data.drop('TBG', axis=1)
thyroid_data = thyroid_data.drop("TBG Measured", axis=1)





thyroid_data.head()


# # Missing Data Preprocessing 
# - Identify missing and abnormal data




def fill_missing_data(dataframe, category_label):
    category = dataframe[category_label]
    cat_missing = category[category != '?']
    cat_val = pd.to_numeric(cat_missing).mean()
    condition = dataframe[category_label]== '?'
    dataframe.loc[condition, category_label] = cat_val





def print_column(dataframe, column):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(dataframe[column])


# ## Age




thyroid_data.loc[thyroid_data['Age'] == '455', 'Age'] = '45'
fill_missing_data(thyroid_data, 'Age')
# print_column(thyroid_data, 'Age')


# ## TSH




fill_missing_data(thyroid_data, 'TSH')
# print_column(thyroid_data, 'TSH')


# ## T3




fill_missing_data(thyroid_data, 'T3')
# print_column(thyroid_data, 'T3')


# ## TT4




fill_missing_data(thyroid_data, 'TT4')
# print_column(thyroid_data, 'TT4')


# ## T4U




fill_missing_data(thyroid_data, 'T4U')
# print_column(thyroid_data, 'T4U')


# ## FTI




fill_missing_data(thyroid_data, 'FTI')
# print_column(thyroid_data, 'FTI')


# ## Sex




thyroid_data = thyroid_data[thyroid_data['Sex'] != '?']





# Last check to ensure no missing data
find_missing_columns(thyroid_data)


# # Separate Columns for Further Processing
# ------




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





continuous_cols, categorical_cols = separate_columns(thyroid_data)


# # Scale Continuous Data
# -----




def normalize_column(dataframe, column):
    dataframe[column] = dataframe[column].apply(pd.to_numeric)
    dataframe[column] = ((dataframe[column]-dataframe[column].min())/
                        (dataframe[column].max()-dataframe[column].min()))





for col in continuous_cols:
    normalize_column(thyroid_data, col)


# # Binarize Categorical Data
# -----




# Binarize Sex column
conditionF = thyroid_data['Sex'] == 'F'
conditionM = thyroid_data['Sex'] == 'M'

thyroid_data.loc[conditionF, 'Sex'] = 0
thyroid_data.loc[conditionM, 'Sex'] = 1





def convert_category(dataframe, column):
    conditionF = dataframe[column] == 'f'
    conditionT = dataframe[column] == 't'

    dataframe.loc[conditionF, column] = 0
    dataframe.loc[conditionT, column] = 1





# Binarize T/F columns
for col in categorical_cols:
    convert_category(thyroid_data, col)


# # Balance Categorical Data
# -------




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




thyroid_data_balanced = dataset_to_sample(thyroid_data, sample=True)





thyroid_data_balanced["Category"].value_counts()





thyroid_data_balanced.to_csv("thyroid_balanced.csv", index=False)


# ### Unbalanced Data




thyroid_data_unbalanced = dataset_to_sample(thyroid_data, sample=False)





thyroid_data_unbalanced["Category"].value_counts()





thyroid_data_unbalanced.to_csv("thyroid_unbalanced.csv", index=False)

