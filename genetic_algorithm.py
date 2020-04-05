#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Loading the reuired libraries
import pandas as pd
import os
import numpy as np
import sys
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree.export import export_text
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt


# In[42]:


def create_multiclass_roc(actual, predicted, classes, split=''):
    y_true_binarized = label_binarize(actual, classes=class_names)
    y_pred_proba     = predicted

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
    roc_auc     = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(split + ': Receiver operating characteristic (Micro)')
    plt.legend(loc="lower right")
    plt.show()


# In[43]:


# Function to pretty print confusion matrix
# Credit : Shay Palachy
# Source : https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #return fig


# In[44]:


# Setting the directory where data resides
#location_of_dataset = os.path.join("../datasets/", 'iris.data')
location_of_dataset = str(sys.argv[1])


# In[45]:


# Loading the data
df_input_2 = pd.read_csv(location_of_dataset, 
                         names = ["sepal length in cm",
                                  "sepal width in cm",
                                  "petal length in cm",
                                  "petal width in cm",
                                  "class"])


# In[46]:


# Setting the predictor & target variables
X  = df_input_2[df_input_2.columns[:-1]]
y  = df_input_2[df_input_2.columns[-1]]


# In[47]:


# Creating train and test split for model building and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=42)


# In[48]:


clf_dt  = DecisionTreeClassifier(criterion='gini', random_state=42) # we have noticed that gini was slightly better 
clf_rdf = RandomForestClassifier(n_estimators=100, random_state=42)

#Fitting both decision tree and RDF model 
clf_dt.fit(X_train, y_train)
clf_rdf.fit(X_train, y_train)


# #### 2.1
# ##### Decision Tree Classifier 

# In[49]:


class_names = clf_dt.classes_
print("Training accuracy: {0}".format(accuracy_score(y_true= y_train, y_pred=clf_dt.predict(X_train))))
print("Testing accuracy: {0}".format(accuracy_score(y_true= y_test, y_pred=clf_dt.predict(X_test))))


# In[50]:


cf = confusion_matrix(y_true=y_train, y_pred=clf_dt.predict(X_train), labels = class_names)
print_confusion_matrix(cf, class_names=class_names)


# In[51]:


cf = confusion_matrix(y_true=y_test, y_pred=clf_dt.predict(X_test), labels = class_names)
print_confusion_matrix(cf, class_names=class_names)


# In[52]:


print(classification_report(y_true=y_train, y_pred=clf_dt.predict(X_train), labels=class_names))
# Note - Sensitivity is same as recall


# In[53]:


print(classification_report(y_true=y_test, y_pred=clf_dt.predict(X_test), labels=class_names))
# Note - Sensitivity is same as recall


# In[54]:


# Train: Recall of negative class is specificity so by the confusion matrix above 
# Iris-versicolor = 1
# Iris-virginica  = 1
# Iris-setosa     = 1


# In[55]:


# Test: Recall of negative class is specificity so by the confusion matrix above 
# Iris-versicolor = 40/41
# Iris-virginica = 41/42
# Iris-setosa = 35/37


# In[56]:


create_multiclass_roc(actual=y_train, predicted = clf_dt.predict_proba(X_train), classes=class_names, split='Train')
create_multiclass_roc(actual=y_test, predicted = clf_dt.predict_proba(X_test), classes=class_names, split='Test')


# ##### Random Forest Classifier

# In[57]:


class_names = clf_rdf.classes_

y_pred = clf_rdf.predict(X_train)
y_true = y_train
print("Train Accuracy: {0}".format(accuracy_score(y_true= y_true, y_pred=y_pred)))

y_pred = clf_rdf.predict(X_test)
y_true = y_test
print("Test Accuracy: {0}".format(accuracy_score(y_true= y_true, y_pred=y_pred)))


# In[58]:


# Training : Confusion matrix
cf = confusion_matrix(y_true=y_train, y_pred=clf_rdf.predict(X_train), labels = class_names)
print_confusion_matrix(cf, class_names=class_names)


# In[59]:


# Testing : Confusion matrix
cf = confusion_matrix(y_true=y_test, y_pred=clf_rdf.predict(X_test), labels = class_names)
print_confusion_matrix(cf, class_names=class_names)


# In[60]:


# Training classficiation report 
print(classification_report(y_true=y_train, y_pred=clf_rdf.predict(X_train), labels=class_names))


# In[61]:


print(classification_report(y_true=y_test, y_pred=clf_rdf.predict(X_test), labels=class_names))
# Note - Sensitivity is same as recall


# In[62]:


# Train: Recall of negative class is specificity so by the confusion matrix above 
# Iris-versicolor = 1
# Iris-virginica  = 1
# Iris-setosa     = 1


# In[63]:


# Test: Recall of negative class is specificity so by the confusion matrix above 
# Iris-versicolor = 40/41
# Iris-virginica = 42/42
# Iris-setosa = 36/37


# In[64]:


create_multiclass_roc(actual=y_train, predicted = clf_rdf.predict_proba(X_train), classes=class_names, split='Train')
create_multiclass_roc(actual=y_test, predicted = clf_rdf.predict_proba(X_test), classes=class_names, split='Test')


# In[65]:


r = export_text(clf_dt, feature_names=list(df_input_2.columns.values[:-1]))
print(r)


# #### 2.2

# There is no need of hyper parameter tunning as with dataset we have, we are able to perfectly classify using Random forest model. We should use Random Forest model as it gives an AUC of 1 for both training and test data. ROC curve and AUC have for both the algorithms have been shown above
