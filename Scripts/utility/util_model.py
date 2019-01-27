#======================================================
# Model Utility Functions
#======================================================
'''
Version 1.0
Utility functions for model building
'''
# Import modules
import os
import copy
import time
import random 
import numpy as np
import pandas as pd
from subprocess import call
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import f1_score, accuracy_score


#------------------------------
# Utility Functions
#------------------------------
def downsample_df(df, labels_df, random_seed):
    num_of_yt   = sum(labels_df)
    random.seed(random_seed+1)
    downsample_bad_ix   = random.sample(np.where(labels_df == 0)[0], num_of_yt)
    good_ix             = np.where(labels_df == 1)[0]
    downsampled_full_ix = np.append(downsample_bad_ix, good_ix)
    df_ds          = pd.concat([df.iloc[[index]] for index in downsampled_full_ix])
    return df_ds

def plot_cfmt(cfmt, classes,
            title='Confusion matrix',
            cmap=plt.cm.Blues,
            save_path=None,
            colorbar=True,
            fontsize=None):
    '''
    This function prints and plots the confusion matrix.
    '''
    plt.imshow(cfmt, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cfmt.max() / 2.
    for i, j in itertools.product(range(cfmt.shape[0]), range(cfmt.shape[1])):
        plt.text(j, i, cfmt[i, j],
                    horizontalalignment="center",
                    color="white" if cfmt[i, j] > thresh else "black", size=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=360)
    else:
        plt.show()

def feature_importance_rf(model, feature_names, verbose=1):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot
    if verbose:
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(feature_names)), importances[indices],
            color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(feature_names)])
        plt.show()

def plot_tree(tree, save_path, feature_names, class_names, dpi=300):
    # Dot path
    dot_save_path = save_path.split('.png')[0] + '.dot'

    # Export as dot file
    export_graphviz(tree, out_file=dot_save_path, 
                    feature_names = feature_names,
                    class_names = class_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    # export_graphviz(tree, out_file=None, max_depth=None, 
    #                 feature_names=None, 
    #                 class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, 
    #                 rotate=False, rounded=False, special_characters=False, precision=3)
    # Convert to png using system command (requires Graphviz)
    
    call(['dot', '-Tpng', dot_save_path, '-o', save_path, '-Gdpi='+str(dpi)])
    os.remove(dot_save_path)