#!/usr/bin/env python
"""
Common data transformation pipelines to ensure we have a single definitive location
where data transformation pipelines are defined to be used in all separate
notebooks.

We give a name to each resulting dataframe we currently support creation of.
For a dataframe named df_labels, there will be a corresponding method in
this module named get_df_labels() which will return the resulting dataframe
after running the raw data through the defined data transformation pipeline.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import mindwandering.features

# A little magic to find the root of this project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(ROOT_DIR, '../..'))

# The location of the data directory, how do we find this as an absolute path?
DATA_DIR = os.path.join(ROOT_DIR, 'data')

## scikit-learn transformer classese used to implement the dataframe
## transformation pipelines
class RenameColumnsUsingMapTransformer(BaseEstimator, TransformerMixin):
    """Use a given map to rename all of the indicated columns.  Also
    as a side effect, columns will be ordered by the order given in
    the map.
    """
    def __init__(self, columns_rename_map):
        self.columns_rename_map = columns_rename_map
        
    def fit(self, df, y=None):
        return self # nothing else to do

    def transform(self, df, y=None):
        df = df.rename(columns = self.columns_rename_map)
        return df

class DropRowsWithEmptyValuesInColumnTransformer(BaseEstimator, TransformerMixin):
    """This transformer will only drop rows
    for the columns that it is asked to check.  And only rows where the value
    in the column is empty or NaN will get dropped.
    """
    def __init__(self, columns_to_check = '[segment_index]'):
        self.columns_to_check = columns_to_check
        
    def fit(self, df, y=None):
        return self # nothing else to do

    def transform(self, df, y=None):
        df = df.dropna(subset = self.columns_to_check)
        return df

class ParticipantIdTransformer(BaseEstimator, TransformerMixin):
    """This transformer expects the participant_id field to have multiple features
    encoded in a string, using '-' as a separator.  It will split out into 2 columns,
    create the location column from the original encoding, and create a unique
    participant id.
    """
    def fit(self, df, y=None):
        return self # nothing else to do

    def transform(self, df, y=None):
        # create a separate dataframe with the two new fields we want
        fields = df.participant_id.str.split('-', expand=True)
        fields.columns = ['BE7', 'participant_id', 'participant_location']
        fields.drop(['BE7'], axis=1, inplace=True)
        
        # map all memphis locations to UM to regularize categorical variable and resulting
        # participant ids
        fields['participant_location'] = fields.participant_location.map({'Memphis': 'UM', 'ND': 'ND'})
        
        # there are duplicate participant ids from the 2 locations.  Map participant id to a string that
        # uses current participant id and the new derived location.  Also the participant id has an initial
        # P which we will remove
        fields['participant_id'] = fields.apply(lambda row: row[0][1:] + '-' + row[1], axis=1)
        
        # replace the participant_id in dataframe to return, add in the participant_location
        df['participant_id'] = fields['participant_id']
        df = df.join(fields['participant_location'])
        
        # new column was added to end, we want it to be at position 1
        cols = df.columns.to_list()
        cols = cols[0:1] + cols[-1:] + cols[1:-1]
        df = df[cols]
        
        return df
    
class TrialDateTimeTransformer(BaseEstimator, TransformerMixin):
    """Transformer to fix the time information in this dataset.  The time
    information was transformed into 2 parts which need to be added together
    to get a valid unix milliseconds (ms) since the epoch result.  
    This transformer combines the fields for start and end time
    into a valid datetime value.  It replaces the start_time and end_time
    fields with the respective datetime values.  It also make the
    trial_length into an int and drops the no longer needed
    time stamp fields.
    """
    def __init__(self, time_field_pairs = [('start_time', 'start_timestamp'), ('end_time', 'end_timestamp')]):
        self.time_field_pairs = time_field_pairs
        
    def fit(self, df, y=None):
        return self # nothing else to do

    def transform(self, df, y=None):
        # iterate through all given pairs of time and timestamp to combine
        for (time, timestamp) in self.time_field_pairs:
            # create a valid datetime value for the pair, replacing the time field with the new datetime value
            df[time] = pd.to_datetime(df[timestamp] + df[time], unit='ms')
            
            # drop the no longer timestamp filed from the dataframe
            df = df.drop(timestamp, axis=1)
       
        return df
    
class SetFeatureTypeTransformer(BaseEstimator, TransformerMixin):
    """Given a list of feature names, and desired type as a list of
    tuple values, transform all features to the indicated data type.
    """
    def __init__(self, feature_type_pairs = [('segment_index', int)]):
        self.feature_type_pairs = feature_type_pairs
        
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # iterate through given pairs of feture name and desired type, converting all indicated
        # features to the new type
        for (feature, new_type) in self.feature_type_pairs:
            # sometimes features have nan, so can only set the type where notna
            #idx = df[feature].notna()
            
            # now set the type for all valid values to the new type
            #df.loc[idx, feature] = df.loc[idx, feature].astype(new_type)
            df[feature] = df[feature].astype(new_type)
            
        return df
    
class CreateMindWanderedLabelTransformer(BaseEstimator, TransformerMixin):
    """Infer a boolean label (False/True) from features that indirectly indicate mind wandering or
    no mind wandering.  Can use either number_of_reports which will be 1 or greater if a mind wandering
    was recorded during the trial.  Also can use first_report_type which is none for all
    trials where no mind wandering occured, and self-caught for all trials where it does.
    """
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        df['mind_wandered_label'] = (df['first_report_type'] == 'self-caught')
        return df

class NumberOfBlinksTransformer(BaseEstimator, TransformerMixin):
    """Number of blinks appear like it should be whole number values, but a number of values have fractional
    parts.  It appears that values between 0 and 1 should actually be a single blink, looking at the mean and min
    and max blink durations.  Thus we need to actually take the ceiling of the number_of_blinks value, then make into
    an int.
    """
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        df['number_of_blinks'] = np.ceil(df.number_of_blinks)
        df['number_of_blinks'] = df.number_of_blinks.astype(int)
        return df

class FillMissingValuesTransformer(BaseEstimator, TransformerMixin):
    """General transformer to fill in missing values for a feature or features with indicated value.
    """
    def __init__(self, feature_value_pairs = [ ('blink_duration_mean', 0.0) ]):
        self.feature_value_pairs = feature_value_pairs
        
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # iterate over all features we are asked to fill with missing values
        for (feature, value) in self.feature_value_pairs:
            df[feature] = df[feature].fillna(value)
        return df

class WinsorizationOutlierTransformer(BaseEstimator, TransformerMixin):
    """This transformer transforms all features of the dataframe to remove outliers.
    It assumes the dataframe has been scaled using standard scaling, such that
    the mean of each feature is 0.0 and the standard deviation is 1.0.
    This transformer scales all features, we might want a more specialized
    one that only scales the requested features, so that you could specify
    which features are already standard scaled.
    """
    def __init__(self, outlier_threshold=3.0):
        self.outlier_threshold = outlier_threshold
        
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # get mean and standard deviation of each feature in the dataframe
        feature_means = df.mean()
        feature_standard_deviations = df.std()
        
        # perform standard scaling on each feature by subtracting the mean and dividing by the standard deviation.
        # the result is a that all features will now have a mean of 0 and a std of 1
        df_outliers = (df.copy() - feature_means) / feature_standard_deviations
        
        # now we can replace outliers that are above/below the outlier_threshold
        df_outliers[df_outliers > self.outlier_threshold] = self.outlier_threshold
        df_outliers[df_outliers < -self.outlier_threshold] = -self.outlier_threshold
  
        # now undo the scaling and return the transformed dataframe
        df_outliers = (df_outliers * feature_standard_deviations) + feature_means

        return df_outliers
    

## access function for the data transformation pipelines
def get_df_raw():
    """This function simply loads the raw data from the source
    data file into a pandas dataframe, and returns it.
    """
    raw_data_file = os.path.join(DATA_DIR, 'mindwandering-raw-data.csv')
    df_raw = pd.read_csv(raw_data_file, sep='\t', lineterminator='\r')
    return df_raw

def get_df_experiment_metadata():
    """The experiment metadata dataframe contains metadata information
    of the experiment.  Information such as the participant id, location,
    time and length of experimental trials, etc.  This metadata information
    also contains the segment id, which defines the temporal sequence
    of the trials done by each participant.
    """
    # we start by creating data frame with the needed columns and renaming them, before
    # any transformation pipelines.
    experiment_metadata_features_map = {
        'ParticipantID':  'participant_id',
        'SegmentIndex':   'segment_index',
        'StartTime(ms)':  'start_time',
        'EndTime(ms)':    'end_time',
        'Length(ms)':     'trial_length',
        'StartTimestamp': 'start_timestamp',
        'EndTimestamp':   'end_timestamp',
    }

    # execute transformation pipeline
    experiment_metadata_pipeline = Pipeline([
        ('rename_columns',          RenameColumnsUsingMapTransformer(experiment_metadata_features_map)),
        ('drop_empty_rows',         DropRowsWithEmptyValuesInColumnTransformer(['segment_index'])),
        ('extract_participant_id',  ParticipantIdTransformer()),
        ('transform_time_values',   TrialDateTimeTransformer([('start_time', 'start_timestamp'), ('end_time', 'end_timestamp')])),
        ('transform_feature_types', SetFeatureTypeTransformer([('segment_index', int), ('trial_length', int)])),
    ])

    df_raw = get_df_raw().copy()
    df_experiment_metadata = experiment_metadata_pipeline.fit_transform(df_raw[experiment_metadata_features_map.keys()])

    # return the experiment metadata dataframe
    return df_experiment_metadata

def get_df_label():
    """The label dataframe contains features that should be used as outputs or predicted labels
    for classifiers built from this data.  In particular, the mind_wandered_label feature of
    this dataframe contains a binary (True / False) label suitable for training a binary
    classifier on this data to detect if mind wandering has occurred or not.
    """
    # we start by creating data frame with the needed columns and renaming them, before
    # any transformation pipelines.
    label_features_map = {
        'NumberOfReports':            'number_of_reports',
        'FirstReportType':            'first_report_type',
        'FirstReportContent':         'first_report_content',
        'FirstReportTimestamp':       'first_report_timestamp',
        'FirstReportTrialTime(ms)':   'first_report_trial_time',
        'FirstReportSegmentTime(ms)': 'first_report_segment_time',
    }

    # execute transformation pipeline
    label_pipeline = Pipeline([
        ('rename_columns',             RenameColumnsUsingMapTransformer(label_features_map)),
        ('drop_empty_rows',            DropRowsWithEmptyValuesInColumnTransformer(['number_of_reports'])),
        ('transform_time_values',      TrialDateTimeTransformer([('first_report_trial_time', 'first_report_timestamp')])),
        #('transform_feature_types',   SetFeatureTypeTransformer([('first_report_segment_time', int)])), # actually cant make int and still have NaN values
        ('create_mind_wandered_label', CreateMindWanderedLabelTransformer()),
    ])

    df_raw = get_df_raw().copy()
    df_label = label_pipeline.fit_transform(df_raw[label_features_map.keys()])

    # return the label dataframe
    return df_label



## The following feature maps are used in several functions that
## transform the data feautres using pipelines, so define in
## module global space.

# the 48 features listed as the eye movement descriptive features in paper we are replicating
eye_movement_descriptive_features_map = {
    'FixDurMed':      'fixation_duration_median',
    'FixDurMean':     'fixation_duration_mean',
    'FixDurSD':       'fixation_duration_standard_deviation',
    'FixDurMin':      'fixation_duration_minimum',
    'FixDurMax':      'fixation_duration_maximum',
    'FixDurRange':    'fixation_duration_range',
    'FixDurSkew':     'fixation_duration_skew',
    'FixDurKur':      'fixation_duration_kurtosis',
    'SacDurMed':      'saccade_duration_median',
    'SacDurMean':     'saccade_duration_mean',
    'SacDurSD':       'saccade_duration_standard_deviation',
    'SacDurMin':      'saccade_duration_minimum',
    'SacDurMax':      'saccade_duration_maximum',
    'SacDurRange':    'saccade_duration_range',
    'SacDurSkew':     'saccade_duration_skew',
    'SacDurKur':      'saccade_duration_kurtosis',
    'SacAmpMed':      'saccade_amplitude_median',
    'SacAmpMean':     'saccade_amplitude_mean',
    'SacAmpSD':       'saccade_amplitude_standard_deviation',
    'SacAmpMin':      'saccade_amplitude_minimum',
    'SacAmpMax':      'saccade_amplitude_maximum',
    'SacAmpRange':    'saccade_amplitude_range',
    'SacAmpSkew':     'saccade_amplitude_skew',
    'SacAmpKur':      'saccade_amplitude_kurtosis',
    'SacVelMed':      'saccade_velocity_median',
    'SacVelMean':     'saccade_velocity_mean',
    'SacVelSD':       'saccade_velocity_sd',
    'SacVelMin':      'saccade_velocity_min',
    'SacVelMax':      'saccade_velocity_max',
    'SacVelRange':    'saccade_velocity_range',
    'SacVelSkew':     'saccade_velocity_skew',
    'SacVelKur':      'saccade_velocity_kurtosis',
    'SacAngAbsMed':   'saccade_angle_absolute_median',
    'SacAngAbsMean':  'saccade_angle_absolute_mean',
    'SacAngAbsSD':    'saccade_angle_absolute_standard_deviation',
    'SacAngAbsMin':   'saccade_angle_absolute_minimum',
    'SacAngAbsMax':   'saccade_angle_absolute_maximum',
    'SacAngAbsRange': 'saccade_angle_absolute_range',
    'SacAngAbsSkew':  'saccade_angle_absolute_skew',
    'SacAngAbsKur':   'saccade_angle_absolute_kurtosis',
    'SacAngRelMed':   'saccade_angle_relative_median',
    'SacAngRelMean':  'saccade_angle_relative_mean',
    'SacAngRelSD':    'saccade_angle_relative_standard_deviation',
    'SacAngRelMin':   'saccade_angle_relative_minimum',
    'SacAngRelMax':   'saccade_angle_relative_maximum',
    'SacAngRelRange': 'saccade_angle_relative_range',
    'SacAngRelSkew':  'saccade_angle_relative_skew',
    'SacAngRelKur':   'saccade_angle_relative_kurtosis',
}

# the 8 pupil diameter descriptive features
pupil_diameter_descriptive_features_map = {
    'PupilDiametersZMed':   'pupil_diameter_median',
    'PupilDiametersZMean':  'pupil_diameter_mean',
    'PupilDiametersZSD':    'pupil_diameter_standard_deviation',
    'PupilDiametersZMin':   'pupil_diameter_minimum',
    'PupilDiametersZMax':   'pupil_diameter_maximum',
    'PupilDiametersZRange': 'pupil_diameter_range',
    'PupilDiametersZSkew':  'pupil_diameter_skew',
    'PupilDiametersZKur':   'pupil_diameter_kurtosis',
}

# The 2 blink features used.  We do not use all of the other derived statistics here because many 
# times number of blinks are 0 or 1 for  atrial, meaning mean, standard deviation, and other measures are not really meaningful.
# There are actually 2260 trials where no blinks occur, and none of these would have meaningful statistics, and of the remaining,
# something like 1191 had a single blink, meaning many statistics like standard deviation don't make sense in those cases.
blink_features_map = {
    'BlinkDurN':     'number_of_blinks',
    'BlinkDurMean':  'blink_duration_mean',
}

# the 4 miscellaneous features used in the results
miscellaneous_features_map = {
    'SacDurN':               'number_of_saccades',
    'horizontalSaccadeProp': 'horizontal_saccade_proportion',
    'FxDisp':                'fixation_dispersion',
    'FxSacRatio':            'fixation_saccade_durtion_ratio',
}

# combine all 4 types of feature dictionaries into a merged dictionary of the 62 features
feature_map = {
    **eye_movement_descriptive_features_map,             
    **pupil_diameter_descriptive_features_map, 
    **blink_features_map, 
    **miscellaneous_features_map
}


def get_df_features():
    """This dataframe contains the basic set of 62 features that were used in the article
    being replicated initially in this project.  The features are extracted from the data
    and cleaned a little to fill in some missing values and fix a few small issues.  But
    this set of features is not scaled or otherwise processed.  All features in
    this dataframe are float64 datatypes or int64 datatypes for numbers that represent
    a count (e.g. number_of_saccades).
    """
    # execute transformation pipeline
    feature_pipeline = Pipeline([
        ('rename_columns',               RenameColumnsUsingMapTransformer(feature_map)),
        ('drop_empty_rows',              DropRowsWithEmptyValuesInColumnTransformer( ['fixation_duration_mean'] )),
        ('transform_number_of_blinks',   NumberOfBlinksTransformer()),
        ('fill_missing_blink_durations', FillMissingValuesTransformer( [('blink_duration_mean', 0.0)] )),
    ])

    # this pipeline runs on the raw features map
    df_raw = get_df_raw().copy()
    df_features = feature_pipeline.fit_transform(df_raw[feature_map.keys()])
    
    # return the features dataframe
    return df_features

def transform_df_features_standard_scaled(df_features):
    """ This method takes a dataframe of features and transforms it using
    standard scaling.  Standard scaling scales all features to have a mean
    of 0 and a standard deviation of 1
    """
    # execute transformation pipeline
    features_standard_scaled_pipeline = Pipeline([
        ('standard_scaler',              StandardScaler()),
    ])

    # this pipeline scales the features using standard scaling for the set of features it is given
    df_features_standard_scaled_nparray = features_standard_scaled_pipeline.fit_transform(df_features.copy())

    # the SciKitLearn preprocessors like StandardScaler seem to transform back into a NumPy array.  We can always make
    # a DataFrame a NumPy array, and vice versa.  Lets put this back into a Pandas DataFrame and put back on the feature
    # labels
    df_features_standard_scaled = pd.DataFrame(df_features_standard_scaled_nparray, 
                                               columns = feature_map.values(), 
                                               index = df_features.index)
    
    # return the dataframe with all features scaled using standard scaling
    return df_features_standard_scaled

def transform_df_features_minmax_scaled(df_features):
    """This method takes a dataframe of features and transforms it using
    min-max scaling. Min-max scaling scales all features to have
    values that range from 0.0 to 1.0.
    """
    # execute transformation pipeline
    features_minmax_scaled_pipeline = Pipeline([
        ('minmax_scaler',              MinMaxScaler()),
    ])

    # this pipeline reuses the results of the standard df_features, and adds standard scaling
    df_features_minmax_scaled_nparray = features_minmax_scaled_pipeline.fit_transform(df_features.copy())

    # the SciKitLearn preprocessors like StandardScaler seem to transform back into a NumPy array.  We can always make
    # a DataFrame a NumPy array, and vice versa.  Lets put this back into a Pandas DataFrame and put back on the feature
    # labels
    df_features_minmax_scaled = pd.DataFrame(df_features_minmax_scaled_nparray, columns = feature_map.values())
    
    # return the dataframe with all features scaled using min-max xcaling
    return df_features_minmax_scaled

def transform_df_features_outliers_removed(df_features, outlier_threshold=3.0):
    """This method takes a dataframe of features and transforms
    it to set outliers to some threshold (Winsorization).
    This method can accept features that are already scaled, or not.
    It will calculate the mean and standard deviation, and 
    Winsorize all features to the threshold given to this function.
    """
    # execute transformation pipeline
    outlier_winsorization_pipeline = Pipeline([
        ('outlier_winsorization', WinsorizationOutlierTransformer(outlier_threshold)),
    ])

    # this pipeline reuses the results of the standard df_features_standard_scaled, and removes outliers using Winsorization
    df_features_outliers_removed = outlier_winsorization_pipeline.fit_transform(df_features.copy())
    
    # return the dataframe after outlier removel
    return df_features_outliers_removed

def transform_df_features_vif_threshold(df_features, feature_ratio=0.52):
    """Calculate the vif scores on the set of features given.  Find a threshold to drop features that
    are above the threshold such that the indicated num_features remains.  We wll do a simple
    binary search to find an appropriate threshold.
    """
    # calculate vif scores for the current set of features
    vif = mindwandering.features.calculate_variance_inflation_factor(df_features)
    
    # perform a search for threshold that will result in the desired number of features
    # should never take more than 20 searches or so to find appropriate threshold
    iter = 0 
    done = False
    min_threshold = 0.0
    max_threshold = 40.0
    mid_threshold = (max_threshold + min_threshold) / 2.0
    df_features_vif = None
    
    # calculate the number of features to keep based on the asked for feature_ratio
    num_trials, num_features = df_features.shape
    num_features_to_keep = int(feature_ratio * num_features)
    
    while not done and iter < 20:
        # test if we found a good threshold and stop when we do
        idxs = (vif < mid_threshold)
        if sum(idxs) == num_features_to_keep:
            df_features_vif = df_features.loc[:,idxs.values].copy()
            done = True
        # otherwise keep searching
        else:
            if sum(idxs) > num_features_to_keep:
                max_threshold = mid_threshold
            else:
                min_threshold = mid_threshold
            mid_threshold = (min_threshold + max_threshold) / 2.0
        
        iter += 1
        
    # df_features is either None if not found, or a new features dataframe was created with
    # the desired number of features
    return df_features_vif

def transform_df_features_correlation_ranking_cutoff(df_features, df_label, participant_ids, feature_ratio=0.5, label_weight=0.5):
    """Calculate feature ranks scores using reference paper describe method of feature cross-correlation,
    and feature to label correlation.  Trim and keep onl the top feature_ratio percentage of features
    as ranked by this method.  
    """
    # get the feature rankings using correlation scores
    corr_scores = mindwandering.features.rank_features_using_correlation_scores(df_features, df_label, participant_ids, label_weight=label_weight)
    
    # use feature_ratio to determine desired number of features
    # get that number of the top best features for the filter
    num_trials, num_features = df_features.shape
    num_features_to_keep = int(num_features * feature_ratio)
    best_features = corr_scores.iloc[:num_features_to_keep].index.tolist()
    
    # filter the df_features input dataframe based on the selected features
    df_features_best = df_features[best_features].copy()
    
    return df_features_best