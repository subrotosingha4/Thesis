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
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler, NearMiss, AllKNN, CondensedNearestNeighbour, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.base import BaseSampler, SamplerMixin

import mindwandering.features

import warnings

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


## access function for the data transformation pipelines.  These are functions
## that give the starting dataframes we always need before any meta-project
## transformations of the features or other information

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


def get_participant_ids():
    """We mostly only ever want the list of participant_id's from the experiment
    metadata (to group cross validation by participants).  So this method specifically
    allows a convenient way to get this information.  It is returned as
    a pandas series, the sinnle participant_id column.
    """
    df_experiment_metadata = get_df_experiment_metadata()
    return df_experiment_metadata.participant_id.copy()


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


def get_mind_wandered_label():
    """Convenience method to get the mind_wandered label from the general dataframe
    label information.  Most (all) tasks where we are building a binary classifier
    will only need the mind_wandered_label, which is a binary target label suitable
    for training binary classifiers.
    """
    df_label = get_df_label()
    return df_label.mind_wandered_label.copy()


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


## Project meta-parameter pipeline transformers.  These transformers can be used
## in project experiment pipelines for a grid search, to transform the data before
## training, e.g. by removing outliers, scaling, using vif thresholding, etc.

class FeatureScalerTransformer(BaseEstimator, TransformerMixin):
    """This transformer will apply feature scaling to a passed in
    data frame.  The transformer supports standard scaling,
    min-max scaling or no scaling.  For no scaling, the transform
    method does not.  No scaling is here to support grid search, to
    allow for project meta-parameter grid points with no scaling
    to be tested.
    """
    def __init__(self, type_of_scaling = 'none'):
        self.type_of_scaling = type_of_scaling
        
    def fit(self, df, y=None):
        return self # nothing else to do

    def transform(self, df, y=None):
        # empty pipeline, will put in type of scaling if any we want
        scaling_pipeline = None
        
        # if no scaling requested, do nothing to transform the data frame
        if self.type_of_scaling == 'none':
            return df
        
        # perform standard scaling if requested
        elif self.type_of_scaling == 'standard':
            # add a standard scaling step into the pipeline
            scaling_pipeline = Pipeline([
                ('standard_scaler', StandardScaler())
            ])
            
        # perform min-max scaling if requested
        elif self.type_of_scaling == 'minmax':
            # add a min-max scaling step into the pipeline
            scaling_pipeline = Pipeline([
                ('minmax_scaler', MinMaxScaler())
            ])

        # otherwise an invalid project metaparamter received, croak
        else:
            raise ValueError('<FeatureScalerTransformer> received unknown type_of_scaling: ' + self.type_of_scaling)
            
        # perform the actual scaling
        array_scaled = scaling_pipeline.fit_transform(df)

        # SciKitLearn StandardScaler returns back pandas dataframe as
        # a NumPy array.  We need it as dataframe still, so convert it back
        # feature_map is a dictionary defined in this data.mindwandering scope
        df_scaled = pd.DataFrame(array_scaled, columns=df.columns, index=df.index)
        
        # return the transformed (or not) dataframe
        return df_scaled


class WinsorizationOutlierTransformer(BaseEstimator, TransformerMixin):
    """This transformer transforms all features of the dataframe to remove outliers.
    It assumes the dataframe has been scaled using standard scaling, such that
    the mean of each feature is 0.0 and the standard deviation is 1.0.
    This transformer scales all features, we might want a more specialized
    one that only scales the requested features, so that you could specify
    which features are already standard scaled.
    
    This transformer supports use in a grid search.  So an outlier_threshold of
    0 is interpreted to mean that no Winsorization should be performed on the
    outliers.
    """
    def __init__(self, outlier_threshold=0.0):
        self.outlier_threshold = outlier_threshold
        
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # only perform Winsorization if the outlier threshold is specified as non zero
        # if it is zero just return the dataframe unmodified
        if self.outlier_threshold < 0.5:
            return df
        
        # otherwise actually perform the outlier transformation
        # get mean and standard deviation of each feature in the dataframe
        feature_means = df.mean()
        feature_standard_deviations = df.std()
        
        # perform standard scaling on each feature by subtracting the mean and dividing by the standard deviation.
        # the result is a that all features will now have a mean of 0 and a std of 1
        df_outliers = (df - feature_means) / feature_standard_deviations
        
        # now we can replace outliers that are above/below the outlier_threshold
        df_outliers[df_outliers > self.outlier_threshold] = self.outlier_threshold
        df_outliers[df_outliers < -self.outlier_threshold] = -self.outlier_threshold
  
        # now undo the scaling and return the transformed dataframe
        df_outliers = (df_outliers * feature_standard_deviations) + feature_means

        return df_outliers
    

class VIFThresholdTransformer(BaseEstimator, TransformerMixin):
    """Calculate the vif scores on the set of features given.  Find a threshold to drop features that
    are above the threshold such that the indicated ratio of features remains in the dataframe.
    We wll do a simplebinary search to find an appropriate threshold.
    
    This pipeline transformer supports grid search.  The feature_ratio must be a value between
    0.0 and 1.0 (a percentage of the desired features that are below the VIF score).  If
    feature_ratio is set to 1.0, then all features are retained (no VIF thresholding occurs).
    """
    def __init__(self, feature_ratio=1.0, score_threshold = 0):
        self.feature_ratio = feature_ratio
        self.score_threshold = score_threshold
        
    def fit(self, df, y=None):
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # special case, just return original df if asked not to do vif threshold.
        # we use 1.0 ratio to indicate we want all features, thus no vif clipping needed.
        if self.feature_ratio >= 1.0 and self.score_threshold == 0:
            return df
        
        # otherwise do vif clipping
        # calculate vif scores for the current set of features
        # need to look into vif calculations, something is not right with the
        # values and divide by 0 runtime warnings we get
        #with warnings.catch_warnings():
        #    warnings.simplefilter('ignore')
        #    vif = mindwandering.features.calculate_variance_inflation_factor(df)
        #    warnings.simplefilter('default')
        vif = mindwandering.features.calculate_variance_inflation_factor(df)
        
        # if selection by the threshold, simply select all features that are below the
        # indicated threshold
        if self.score_threshold > 0:
            #idxs = (vif > self.score_threshold)
            cols = df.columns[vif > self.score_threshold]
            #df_features_vif = df.loc[:,idxs].copy()
            df_features_vif = df.drop(cols, axis=1)
            
        # otherwise perform a search for threshold that will result in the desired number of features
        # should never take more than 20 searches or so to find appropriate threshold
        elif self.feature_ratio < 1.0:
            iter = 0 
            done = False
            min_threshold = 0.0
            vif = np.nan_to_num(vif)
            vif[vif > 500] = 500
            max_threshold = 2.0 * vif.max()
            mid_threshold = (max_threshold + min_threshold) / 2.0
            df_features_vif = None

            # calculate the number of features to keep based on the asked for feature_ratio
            num_trials, num_features = df.shape
            num_features_to_keep = int(self.feature_ratio * num_features)

            while not done and iter < 100:
                # test if we found a good threshold and stop when we do
                idxs = (vif < mid_threshold)
                #print(min_threshold, mid_threshold, max_threshold, sum(idxs))
                if sum(idxs) == num_features_to_keep:
                    df_features_vif = df.loc[:,idxs].copy()
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
        if df_features_vif is None:
            raise ValueError('<VIFThresholdTransformer> no features selected from vif scaling, feature_ratio: ' + self.feature_ratio)
            
        return df_features_vif

   
class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """This transformer pipeline can/will support multiple types of feature selection on the given dataframe.
    
    We implement a custom feature selection method using correlation scores between the features, and between each
    feature and the output label.  Thus this transformer needs not only the features, but the labels, and
    the participant_ids index (for sampling by participants when determining features).  The build of the
    work in calculating the correlation scores is done by the mindwandering.features.rank_features_using_correlation_scores.
    
    This pipeline transformer supprts grid search over featuer selection as a project meta-parameter.  Thus requesting
    type_of_selection of 'none' will result in no feature selection taking place.
    
    NOTE: you shouldn't use 'none' in combination with feature_ratios, as you will then get grid searches that
    don't do feature selection multiple times for each desired feature_ratio.  Instead specify a feature_ratio of 1.0
    so that no features are selected by the criteria.
    """
    def __init__(self, df_label, participant_ids, label_weight=0.5, type_of_selection='none', feature_ratio=1.0):
        self.df_label = df_label
        self.participant_ids = participant_ids
        self.label_weight = label_weight
        self.type_of_selection = type_of_selection
        self.feature_ratio = feature_ratio
        
    def fit(self, df, y=None):
        self.df_label = y
        return self # nothing else to do
    
    def transform(self, df, y=None):
        # if no feature selection requested, or if the ratio is 1.0,
        # do nothing to transform the data frame by selecting features, keep all of the featuress
        if self.type_of_selection == 'none' or self.feature_ratio >= 1.0:
            return df
        
        # otherwise perform project specific correlation selection if requested to do so
        elif self.type_of_selection == 'correlation-selection':
            # get the feature rankings using correlation scores
            corr_scores = mindwandering.features.rank_features_using_correlation_scores(df, self.df_label, self.participant_ids, label_weight=self.label_weight)

            # use feature_ratio to determine desired number of features
            # get that number of the top best features for the filter
            num_trials, num_features = df.shape
            num_features_to_keep = int(num_features * self.feature_ratio)
            best_features = corr_scores.iloc[:num_features_to_keep].index.tolist()
            
            # filter the df_features input dataframe based on the selected features
            df_best_features = df[best_features]
            return df_best_features
        
        # or perform tree based selection if asked to.
        elif self.type_of_selection == 'tree-selection':
            # create an ensemble tree classifier and trait it on these features
            clf = ExtraTreesClassifier(n_estimators=500)
            clf.fit(df, self.df_label)
            
            # use tree feature importance to do the feature selection
            num_trials, num_features = df.shape
            num_features_to_keep = int(num_features * self.feature_ratio)
            tree_scores = pd.Series(data=clf.feature_importances_, index=df.columns)
            tree_scores = tree_scores.sort_values(ascending=False)
            best_features = tree_scores.iloc[:num_features_to_keep].index.tolist()
            df_best_features = df[best_features]
            return df_best_features
            
        # received an invalid/unknown type of feature selection request   
        else:
            raise ValueError('<FeatureSelectionTransformer> unknown feature selection method requested: ' + self.type_of_selection)


class ClassImbalanceTransformer(BaseSampler, SamplerMixin):
    """This transformer pipeline can/will support multiple types of class imbalance up/down sampling.  This is a wraper
    around some imbalance library classes (RandomUnderSampler and SMOTE), so that we can select different types of
    balancing within a GridSearch.  As usual we also support a 'none' parameter, so that the grid search can specify no
    class balancing as a grid point of the search.
    """
    def __init__(self, balancer_type='none', sampling_strategy='auto'):
        super().__init__()
        self.balancer_type = balancer_type
        self.sampling_strategy = sampling_strategy
        self._sampling_type = 'ensemble'

    def _fit_resample(self, df, label):
        # if no balancing requested, return the df untouched
        if self.balancer_type == 'none':
            return df, label
        
        sampler = None
        
        # perform random undersampling if requested
        if self.balancer_type == 'random-undersampler':
            # add a standard scaling step into the pipeline
            sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy)
        
        # perform all knn undersampling
        elif self.balancer_type == 'allknn':
            # add a standard scaling step into the pipeline
            sampler = AllKNN(sampling_strategy=self.sampling_strategy)
            
        # perform near miss undersampling
        elif self.balancer_type == 'nearmiss':
            # add a standard scaling step into the pipeline
            sampler = NearMiss(sampling_strategy=self.sampling_strategy)

        # perform condensed nn
        elif self.balancer_type == 'condensed-nn':
            # add a standard scaling step into the pipeline
            sampler = CondensedNearestNeighbour(sampling_strategy=self.sampling_strategy)

        # perform neighbourhood cleaning rule
        elif self.balancer_type == 'neighbourhood-cleaning':
            # add a standard scaling step into the pipeline
            sampler = CondensedNearestNeighbour(sampling_strategy=self.sampling_strategy)

        # perform one sided selection
        elif self.balancer_type == 'one-sided-selection':
            # add a standard scaling step into the pipeline
            sampler = OneSidedSelection(sampling_strategy=self.sampling_strategy)

        # perform random oversampling
        elif self.balancer_type == 'random-oversampler':
            # add a standard scaling step into the pipeline
            sampler = RandomOverSampler(sampling_strategy=self.sampling_strategy)

        # perform smote oversampling / data augmentation
        elif self.balancer_type == 'smote':
            # add a standard scaling step into the pipeline
            sampler = SMOTE(sampling_strategy=self.sampling_strategy)
        
        # perform smote w/ tomek cleaning
        elif self.balancer_type == 'smote-tomek':
            # add a standard scaling step into the pipeline
            sampler = SMOTETomek(sampling_strategy=self.sampling_strategy)
        
        # perform smote w/ enn cleaning
        elif self.balancer_type == 'smote-enn':
            # add a standard scaling step into the pipeline
            sampler = SMOTEENN(sampling_strategy=self.sampling_strategy)
        
        else:
           raise ValueError('<ClassImbalanceTrans> unknown feature selection method requested: ' + self.balancer_type)
        
        # perform the actual up or down sampling and return the transformed dataframe
        df_balanced, label_balanced = sampler.fit_resample(df, label)
        return df_balanced, label_balanced


class GridSearchProgressHack(BaseEstimator, TransformerMixin):
    """Looking for a better way to at least get the count of the number of fits so far.  At beginning of a long
    grid search, you know how many candidates and fits will need to be performed.  But verbose output from 
    GridSearch doesn't indicate which candidate/fit on or like a 1/N type progress, which makes it hard to determine
    how far along you have progressed.
    
    This is a hack until can find something better.  But simply add this transformer to the beginning of the
    pipeline, and it will at list print out the number of fits so far
    """
    num_fits = 1
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def fit(self, df, y=None):
        if self.verbose:
            print('Fit %03d:' % GridSearchProgressHack.num_fits, end=' ')
        else:
            print('%04d' % GridSearchProgressHack.num_fits, end=' ')
        GridSearchProgressHack.num_fits += 1
        return self # nothing else to do
    
    def transform(self, df, y=None):
        return df