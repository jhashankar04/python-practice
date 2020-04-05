#!/usr/bin/env python
# coding: utf-8

# In[54]:


# Loading the reuired libraries
import pandas as pd
import os
import sys
import numpy as np
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


# In[55]:


# Setting the directory where data resides
#location_of_dataset = os.path.join("../datasets/", 'wifi_localization.txt')
location_of_dataset = str(sys.argv[1])


# In[56]:


# Loading the data
df_input_1 = pd.read_csv(location_of_dataset, sep = "\t", header=None, prefix='col_')


# In[57]:


# Setting the predictor & target variables
X  = df_input_1[df_input_1.columns[:-1]]
y  = df_input_1[df_input_1.columns[-1]]


# In[58]:


# Creating train and test split for model building and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)


# In[59]:


def compute_entropy(target):
    return sum(target.value_counts(normalize = True).apply(lambda x: -1*x*log2(x)))

def get_best_feature_and_cut_off(df, 
                                 target = 'col_7', 
                                 columns = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6'], 
                                 process_so_far = ''):
    
    """
    This function gets best feature and cut-off to split the data with an objective of 
    maximizing information gain 
    """
    print('processing subset: {0}'.format(process_so_far))
    
    if df[target].nunique() != 1:
        
        entropy_parent = compute_entropy(df[target])
        best_ig      = 0 
        best_col     = None 
        best_cut_off = None
        best_ent     = None
        for col in columns: 
            for cut_off in df[col].unique(): 
                temp_var = (df[col] <= cut_off).astype('int')
                result = 0
                for value in temp_var.unique():
                    subset = temp_var == value
                    result+=(compute_entropy(df.loc[subset,target])*sum(subset))/df.shape[0]
                current_ig = entropy_parent - result
                if current_ig > best_ig: 
                    best_col = col 
                    best_cut_off = cut_off 
                    best_ig = current_ig
                    best_ent = entropy_parent

        print("Best column is {0} at {1} cut-off. IG is {2} & entropy is {3}".format(best_col, 
                                                                                     best_cut_off, 
                                                                                     best_ig, 
                                                                                     best_ent)) 
        if process_so_far == '':                                                                  
            to_return = ('status: not_complete', 
                         {'best_col': best_col, 
                        'cut_off' : str(best_cut_off), 
                        'left_child'  : "df['{0}'] <={1}".format(best_col, best_cut_off), 
                        'right_child' : "df['{0}'] >{1}".format(best_col, best_cut_off)}, 
                         df[target].mode()[0])
        else:
            to_return = ('status: not_complete', 
                         {'best_col': best_col, 
                        'cut_off' : str(best_cut_off), 
                        'left_child'  :  '(' + process_so_far + ')' + "& (df['{0}'] <={1})".format(best_col, 
                                                                                                   best_cut_off), 
                        'right_child' :  '(' + process_so_far + ')' + "& (df['{0}'] >{1})".format(best_col, 
                                                                                                  best_cut_off) },
                         df[target].mode()[0])
    else: 
        to_return = ('status: complete', process_so_far, df[target].mode()[0])
 
    return to_return

def id3(df = df_input_1, 
        target='col_7', 
        columns = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']):
    
    """
    This function implements the ID3 algorithm using the best feature & split mechanism 
    """
    
    to_process = [] 
    outcome = get_best_feature_and_cut_off(df)
    print(outcome)
    to_process.extend([outcome[1]['left_child'], outcome[1]['right_child']])
    
    i = 0
    paths = []
    while len(to_process): 
        #1. subset_data
        #2. get best feature & cutoff 
        #3. Add to process list 
        output = get_best_feature_and_cut_off(df[eval(to_process[i])], process_so_far=to_process[i])
        to_process.remove(to_process[i])
        if output[0] != 'status: complete':
            to_process.extend([output[1]['left_child'], output[1]['right_child']])
            print(output[0])
        else:
            decision_path = {'path': output[1], 'class': output[2]}
            paths.append(decision_path) 
            
        print("The length of process list is:" + str(len(to_process)))
        #print(to_process)
    print('completed creating the decision tree')
    return paths

def score(model_obj, df): 
    df['predicted_class'] = ''
    for decison in model_obj:
        df.loc[eval(decison['path']),'predicted_class'] = decison['class']
    return df


# In[60]:


# Function to pretty print confusion matrix
# Credit : Shay Palachy
# Source : https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823

import seaborn as sns
import matplotlib.pyplot as plt

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


# In[61]:


d_tree = id3(df=pd.concat([X_train, y_train], axis = 1))


# In[62]:


df_prediction = score(model_obj=d_tree, df=pd.concat([X_test,y_test], axis = 1))


# In[63]:


y_true = df_prediction['col_7']
y_pred = df_prediction['predicted_class']
cf = confusion_matrix(y_true = y_true, y_pred = y_pred)
#pd.crosstab(df_prediction['col_7'], df_prediction['predicted_class']).reset_index()


# In[35]:


print_confusion_matrix(cf, class_names=[1,2,3,4])


# In[36]:


print("Accuracy: {0}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))
print("F1-score (micro): {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average="micro")))
print("F1-score (macro): {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average="macro")))
print("F1-score for each class : {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average=None)))


# In[46]:


print(classification_report(y_true=y_true, y_pred=y_pred))


# #### Analysis of model metric 
# F1 score is useful when there is class imbalance and we are interested in reduction of false postives & false negatives. Just going by the class balance accuracy is a good enough measure in this scenario

# ### Question 1.2

# In[47]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=42)
clf = clf.fit(X_train, y_train)


# In[48]:


y_true = y_test
y_pred = clf.predict(X_test)


# In[49]:


cf = confusion_matrix(y_true = y_true, y_pred = y_pred)
print_confusion_matrix(cf, class_names=clf.classes_)


# In[50]:


print("Accuracy: {0}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))
print("F1-score (micro): {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average="micro")))
print("F1-score (macro): {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average="macro")))
print("F1-score for each class : {0}".format(f1_score(y_true=y_true, y_pred=y_pred, average=None)))


# In[51]:


print(classification_report(y_true=y_true, y_pred=y_pred))


# In[52]:


# we will compare the predictions of both the above models 
df_outcomes = pd.DataFrame({'y_true':y_true, 'ID3':df_prediction['predicted_class'], 'gini':clf.predict(X_test)})


# In[53]:


# Note that there are 9 miss-matches between the both the prediction
len_mm = X_test[df_outcomes['ID3'] != df_outcomes['gini']].shape[0]
df_analysis = df_outcomes[df_outcomes['ID3'] != df_outcomes['gini']]
print(len_mm)


# ##### Comparison of both the measures (Given the prediction from both are not the same)

# In[43]:


print("Correctness of gini: {0}".format((1-sum(df_analysis['y_true'] != df_analysis['gini'])/len_mm)*100))
print("Correctness of ID3: {0}".format((1-sum(df_analysis['y_true'] != df_analysis['ID3'])/len_mm)*100))


# Overall Gini is slightly better than ID3 when used as impurity metric (see the correctness when prediction in both cases are not the same). ID3 is making mistake in correctly classifying class '3' majority times while gini is incorrectly classifying class '4' out of missmatched predictions between both
