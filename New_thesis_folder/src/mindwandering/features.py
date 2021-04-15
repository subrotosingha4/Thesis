#!/usr/bin/env python
"""
Methods developed for feature selection using the mindwandering dataset of this project.
Provide methods based on reference paper description to try and replicate feature
selection processes used.
"""
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

## The following functions are hand-made recreations of described
## feature selection process that is based on correlation of features
## with the label and cross correlation of features with the other
## features

def get_random_sample_of_participants(participant_ids, sample_ratio = 0.66):
    """We expect a list or series of participant ids.  This method selects
    the sample_size proportion of the participants at random, and returns this
    sampled list.
    """
    num_participants = len(participant_ids)
    num_samples = int(num_participants * sample_ratio)
    return random.sample(participant_ids, num_samples)

def calculate_feature_correlation_scores(df_features, df_label, label_weight=0.5):
    """This function will generate a correlation score for a dataframe of features and a series of the
    target labels.  The score is a combination of how correlated each feature is with the label, and how
    correlated each feature is with other features.  
    
    We calculate all correlations using standard pearson correlation.  The correlation of each feature with
    the output label is a single score.  We take the absolute value of this correlation, as it is the
    magnitude of the correlation that is of interest.  We do a pairwise correlation of each feature with 
    all other features, and then determine the maximum correlation (that is not self-correlation).
    Again we use the absolute value of this maximum correlation.  However, while high correlation with
    the label is good, high correlation with other features is bad.  So we invert the correlation score
    with other features (1 - feature_correlation).  Then we take the weighted average of the label
    correlation and feature correlation.
    
    The result is a single score between 0 and 1 for each feature.  We return this result as a pandas
    series, with the feature labels as index (in original order given in input dataframe), and their
    calculated combined feature correlation score as the value.
    """
    # first calculate correlation score of features with the label, convert to numpy array, and take
    # absolute value.  Result is a vector of the absolute value of the correlation with the label
    label_corr = df_features.corrwith(df_label).fillna(0)
    label_corr = label_corr.to_numpy()
    label_corr = np.absolute(label_corr)
    
    # now calculate cross correlation of each feature with all the others.  Result is an nxn square
    # array of correlation values.  Values in diagnol will be 1 because of self-correlation.
    # replace diagnol by -1 and take the max of each row to get maximum correlation of each feature.
    # Take absolute value before max, as we want the magnitude of the correlation
    feature_corr = df_features.corr().fillna(0).to_numpy()
    feature_corr = np.absolute(feature_corr)
    np.fill_diagonal(feature_corr, -1)
    feature_corr = feature_corr.max(axis=1)
    
    # now combine correlations using a weighted average
    combined_corr = (label_weight * label_corr) + ((1.0 - label_weight) * feature_corr)
    
    # now convert back to a pandas series
    combined_corr_series = pd.Series(data=combined_corr, index=df_features.columns)
    
    return combined_corr_series

def rank_features_using_correlation_scores(df_features, df_label, participant_ids, N=5, label_weight=0.5, sample_ratio=0.66):
    """Perform the complete procedure to rank features using some type of correlation score measure.  We assume
    a function that will calculate correlation scores of features, given a dataframe of the features and a dataframe
    of the labels.
    
    This method performs N=5 (by default) samples of the features.  Samples are drawn by subject, where 66% of
    subjects are sampled each trial.  We combine the calculated feature scores by taking the average.
    We then rank the features (sort them), and return this resulting series of ranked features.
    """
    # a series to accumulate and calculate the final result to return
    num_trials, num_features = df_features.shape
    corr_scores_result = pd.DataFrame(data=np.zeros((num_features,N)), index=df_features.columns)
    
    # make sure we are using the set of participant ids for the features
    # we were given.  Because of cross validation we might only receive
    # some of the trials, so only rank those trials and thos participants
    if len(df_features.index) != len(participant_ids.index):
        participant_ids_for_features = participant_ids.loc[df_features.index]
        df_label_for_features = df_label.loc[df_features.index]        
    else:
        participant_ids_for_features = participant_ids
        df_label_for_features = df_label
        
    # get list of participant ids to use in sampling loop
    unique_participant_ids = participant_ids_for_features.unique().tolist()
    
    for sample_num in range(N):
        # draw a sample of participants
        participant_sample = get_random_sample_of_participants(unique_participant_ids, sample_ratio)
        idx = participant_ids_for_features.isin(participant_sample)
        feature_sample = df_features[idx]
        label_sample = df_label_for_features[idx]
        
        # calculate the feature correlation scores for this randomly drawn sample
        corr_scores = calculate_feature_correlation_scores(feature_sample, label_sample)
        
        # add the scores to the result
        corr_scores_result.iloc[:,sample_num] = corr_scores
        
    # add new column of the average feature correlation scores
    corr_scores_result['mean'] = corr_scores_result.mean(axis=1)
        
    # sort by the mean to get final ranking
    corr_scores_result = corr_scores_result.sort_values(by=['mean'], ascending=False)
    
    return corr_scores_result

def rank_features_using_trees(df_features, df_label, n_estimators=500):
    """An alternative method for feature ranking / selection.  As with previous
    function, we will return a Series/DataFrame with the feature name as index
    and the score/rankings as a value, presorted in descending order.
    
    We use the example tree-based feature selection from scikit-learn 0.24.1 documentation
    [tree-based feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel)
    """
    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    clf = clf.fit(df_features, df_label)
    tree_scores = pd.Series(data=clf.feature_importances_, index=df_features.columns)
    tree_scores = tree_scores.sort_values(ascending=False)
    return tree_scores

def calculate_variance_inflation_factor(df_features):
    """Given a set of features, calculate the
    variance inflation factor (VIF) of the features given
    in the df_features DataFrame.
    
    We use [reference](https://www.statology.org/how-to-calculate-vif-in-python/)
    as source of the calculation, and actually this method really
    is just a wraper around the statsmodel implementation
    of VIF.
    """
    # construct the linear regression design matrices
    # make a copy of the features dataframe and add an intercept column feature as
    # first column, filled with 1'2
    num_trials, num_features = df_features.shape
    df = df_features.copy()
    df.insert( loc=0, column='intercept', value=np.ones(num_trials) )
    features_matrix = df.values
    
    # calculate the variance inflaction factor
    vif_results = [variance_inflation_factor(features_matrix, i) for i in range(num_features + 1)]
    vif = pd.Series(index = df.columns, data=vif_results)
    
    # we aren't interested in the intercept inflation factor for this feature trimming function,
    # so remove it
    vif = vif.iloc[1:]
    
    return vif

