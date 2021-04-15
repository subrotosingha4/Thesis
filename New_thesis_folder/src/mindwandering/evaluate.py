#!/usr/bin/env python
"""
Methods developed for model evaluation.  Convenience methods so we don't copy/paste
the code in multiple places to evaluate accuracy, precision, recall, auc-roc scores and
accompanying figures.
"""
# globally useful imports of standard libraries needed in this library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# metric functions from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def evaluate_model_results(df_label, predictions, probabilities):
    """Given the correct labels, and a set of predictions, evaluate the model.  We
    also need the classifier probabilities, the probability that it is the
    positive (true or 1) class.
    
    We calculate and display accuracy, precision, recall, AUC-ROC score, etc.
    We also plot the precision/recall curve and the AUC-ROC curve.
    """
    # display scores
    accuracy = accuracy_score(df_label, predictions)
    recall = recall_score(df_label, predictions, zero_division=0)
    precision = precision_score(df_label, predictions, zero_division=0)
    aucroc = roc_auc_score(df_label, predictions)

    print('Accuracy score:        ', accuracy)
    print('Recall (TPR) score:    ', recall)
    print('Precision score:       ', precision)
    print('AUC-ROC score:         ', aucroc)
    print('')
    
    # display confusion matrix properties
    cm = confusion_matrix(df_label, predictions)

    # extract tp, tn, fp, fn
    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]

    # true positive rate and false positive rate
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    # specificity
    specificity = tn / (tn + fp)

    print('True positives:  ', tp)
    print('True negatives:  ', tn)
    print('False positives: ', fp)
    print('False negatives: ', fn)
    print('True Positive Rate (recall): ', tpr)
    print('False Positive Rate:         ', fpr)
    print('Specificity:                 ', specificity)
    print('')
    
    print(cm)
    
    # plot the confusiion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot();    
    
    # plot the precision/recall curve
    precision, recall, thresholds = precision_recall_curve(df_label, probabilities)

    plt.figure(figsize=(12,8))
    plt.plot(recall[1:], precision[1:], label='Model precision vs. recall')
    plt.legend()
    plt.xlabel('Recall (Positive label: 1)')
    plt.ylabel('Precision (Positive label: 1)');
    
    # plot the AUC-ROC curve
    num_trials = df_label.shape[0]
    random_probabilities = np.zeros(num_trials)
    random_fpr, random_tpr, _ = roc_curve(df_label, random_probabilities)
    model_fpr, model_tpr, _ = roc_curve(df_label, probabilities)

    plt.figure(figsize=(12,8))
    plt.plot(random_fpr, random_tpr, label='Random performance')
    plt.plot(model_fpr, model_tpr, label='Model performance')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate');    