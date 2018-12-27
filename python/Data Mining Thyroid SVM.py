import itertools

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.svm import LinearSVC

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data
# ----




def get_data(balanced=False):
    if balanced:
        return pd.read_csv("thyroid_balanced.csv",index_col=False)
    else:
        return pd.read_csv("thyroid_unbalanced.csv",index_col=False)


# # Prepare Data For Model
# ------




thyroid_data = get_data(balanced=True)





thyroid_data.head()





targets = thyroid_data['Category'].unique()





X = thyroid_data.drop("Category", axis=1)
y = thyroid_data["Category"]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)





# kf = StratifiedKFold(n_splits=3)

# for train_index, test_index in kf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# # Multiclass Classification with Support Vector Machines
# -------




svm_model = LinearSVC()





svm_model.fit(X_train,y_train)





y_pred = svm_model.predict(X_test)





print("Accuracy:\n", metrics.accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))





unique, counts = np.unique(y_test, return_counts=True)





unique, counts





def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(12,6))
plot_confusion_matrix(cnf_matrix, classes=['hyperthyroid','hypothyroid', 'negative', 'sick'])





precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))





def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(score(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                center=0,
                cmap='Accent',
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax)

plot_classification_report(y_test, y_pred)

