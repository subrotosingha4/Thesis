#!/usr/bin/env python
"""
Unit tests for the functions and transformer pipelines in the mindwandering project
module.
"""
import numpy as np
import unittest
from sklearn.pipeline import Pipeline
from mindwandering.data import get_df_raw, get_participant_ids, get_df_experiment_metadata, get_mind_wandered_label, get_df_label, get_df_features
from mindwandering.data import FeatureScalerTransformer
from mindwandering.data import WinsorizationOutlierTransformer
from mindwandering.data import VIFThresholdTransformer
from mindwandering.data import FeatureSelectionTransformer


# information about cleaned data we expect to see if cleaning transformations
# are working
CLEAN_NUM_TRIALS = 4076
CLEAN_NUM_FEATURES = 62
CLEAN_NUM_PARTICIPANTS = 135
CLEAN_NUM_NEGATIVE_LABELS = 2963
CLEAN_NUM_POSITIVE_LABELS = 1113

# tests of the basic get dataframe methods for the project
class TestGetDF(unittest.TestCase):
    def test_get_df_raw(self):
        """Test that the raw dataframe always has the expected
        properties upon reading it in, such as number of trials, number of features,
        number of participants, and a few other summary features of some columns.
        """
        df = get_df_raw()
        self.assertEqual(df.shape, (4078,129))
        
        participants = df.ParticipantID.unique()
        self.assertEqual(participants.shape, (136,))

    def test_mind_wandered_label(self):
        """Test that the label data transformations are working correctly.
        """
        mind_wandered_label = get_mind_wandered_label()
        self.assertEqual(mind_wandered_label.shape, (CLEAN_NUM_TRIALS,))

        num_negative_labels = sum(mind_wandered_label == False)
        self.assertEqual(CLEAN_NUM_NEGATIVE_LABELS, num_negative_labels)
        num_positive_labels = sum(mind_wandered_label == True)
        self.assertEqual(CLEAN_NUM_POSITIVE_LABELS, num_positive_labels)

    def test_get_df_label(self):
        """Test that the label data transformations are working correctly.
        """
        df = get_df_label()
        self.assertEqual(df.shape, (CLEAN_NUM_TRIALS, 6))
        
        num_negative_labels = sum(df.mind_wandered_label == False)
        self.assertEqual(CLEAN_NUM_NEGATIVE_LABELS, num_negative_labels)
        num_positive_labels = sum(df.mind_wandered_label == True)
        self.assertEqual(CLEAN_NUM_POSITIVE_LABELS, num_positive_labels)
        
        unique_num_reports = df.number_of_reports.unique()
        unique_num_reports.sort()
        self.assertTrue(np.array_equal(unique_num_reports, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) ))
        self.assertEqual(sum(df.number_of_reports == 0), CLEAN_NUM_NEGATIVE_LABELS)
        self.assertEqual(sum(df.number_of_reports > 0), CLEAN_NUM_POSITIVE_LABELS)
        
        unique_first_report_type = df.first_report_type.unique()
        unique_first_report_type.sort()
        self.assertTrue(np.array_equal(unique_first_report_type, np.array(['none', 'self-caught']) ) )
        self.assertEqual(sum(df.first_report_type == 'none'), CLEAN_NUM_NEGATIVE_LABELS)
        self.assertEqual(sum(df.first_report_type == 'self-caught'), CLEAN_NUM_POSITIVE_LABELS)
 
        unique_first_report_content = df.first_report_content.unique()
        unique_first_report_content.sort()
        self.assertTrue(np.array_equal(unique_first_report_content, np.array(['none', 'other', 'task-related']) ) )
        self.assertEqual(sum(df.first_report_content == 'none'), CLEAN_NUM_NEGATIVE_LABELS)
        self.assertEqual(sum(df.first_report_content == 'other') + sum(df.first_report_content == 'task-related'), 
                                                                    CLEAN_NUM_POSITIVE_LABELS)
        
    def test_get_participant_ids(self):
        """Test that participant ids are fetch correctly.
        """
        participant_ids = get_participant_ids()
        self.assertEqual(participant_ids.shape, (CLEAN_NUM_TRIALS, ))
        
        unique_participant_ids = participant_ids.unique()
        self.assertEqual(unique_participant_ids.shape, (CLEAN_NUM_PARTICIPANTS, ))

    def test_get_experiment_metadata(self):
        """Test that the experiment metadata dataframe is being fetched
        correctly.
        """
        df = get_df_experiment_metadata()
        self.assertEqual(df.shape, (CLEAN_NUM_TRIALS, 6))
        
        unique_participant_ids = df.participant_id.unique()
        self.assertEqual(unique_participant_ids.shape, (CLEAN_NUM_PARTICIPANTS, ))

        unique_participant_locations = df.participant_location.unique()
        self.assertEqual(unique_participant_locations.shape, (2, ))
        
        unique_segment_indexes = df.segment_index.unique()
        self.assertEqual(unique_segment_indexes.shape, (57, ))
        
        self.assertEqual(df.trial_length.dtype, int)
        self.assertEqual(df.segment_index.dtype, int)
        self.assertEqual(df.start_time.dtype, 'datetime64[ns]')
        self.assertEqual(df.end_time.dtype, 'datetime64[ns]')

    def test_get_df_features(self):
        """Test cleaned base data frame for project.  Make sure we test that the
        data cleaning transformations work for the base df_features dataframe
        that is created and returned by this module.
        """
        df = get_df_features()
        self.assertEqual(df.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # there should be no missing values in this cleaned data
        self.assertEqual(df.isna().sum().sum(), 0)
        
        # we cleaned number of blinks slightly, had to ceil some numbers to what looked like correct value
        self.assertEqual(df.number_of_blinks.dtype, int)
        unique_number_of_blinks = df.number_of_blinks.unique()
        unique_number_of_blinks.sort()
        self.assertTrue( np.array_equal(unique_number_of_blinks, np.array([0, 1, 2, 3, 4, 5, 6]) ) )
        
        # we filled 2260 missing blink duration means with 0, they were missing because number of
        # blinks were 0
        self.assertEqual( sum(df.number_of_blinks == 0), 2260 )
        self.assertEqual( sum(df.blink_duration_mean == 0.0), 2260 )

        
class TestFeatureScalerTransformer(unittest.TestCase):
    def test_no_feature_scaling(self):
        """Transformer supports grid search by allowing a type of scaling to be none.
        In that case, the dataframe should be returned with no transformations.
        """
        # don't scale the data
        pipeline = Pipeline([
            ('no_feature_scaling', FeatureScalerTransformer(type_of_scaling='none') ),
        ])
        df = get_df_features()
        df_scaled = pipeline.fit_transform(df)
        self.assertTrue(df is df_scaled)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_scaled is df_original)
        self.assertTrue(df_scaled.equals(df_original))
        
    def test_standard_feature_scaling(self):
        """Test that standard scaling seems to work as expected.
        """
        # apply standard scaling
        pipeline = Pipeline([
            ('standard_scaling', FeatureScalerTransformer(type_of_scaling='standard') ),
        ])
        df = get_df_features()
        df_scaled = pipeline.fit_transform(df)
    
        self.assertFalse(df_scaled is df)
        self.assertFalse(df_scaled.equals(df))
        self.assertEqual(df_scaled.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))
        
        # scaled means should all be (close to) 0.0
        self.assertTrue( np.all(np.isclose(df_scaled.mean().to_numpy(), 0.0)) )
    
        # scaled std should all be (close to) 1.0
        self.assertTrue( np.all(np.isclose(df_scaled.std().to_numpy(), 1.0, atol=1e-03)) )
    
    def test_minmax_feature_scaling(self):
        """Test that min/max feature scaling seems to work as expected.
        """
        # apply standard scaling
        pipeline = Pipeline([
            ('standard_scaling', FeatureScalerTransformer(type_of_scaling='minmax') ),
        ])
        df = get_df_features()
        df_scaled = pipeline.fit_transform(df)
    
        self.assertFalse(df_scaled is df)
        self.assertFalse(df_scaled.equals(df))
        self.assertEqual(df_scaled.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # scaled maxes should all be (close to) 1.0
        self.assertTrue( np.all(np.isclose(df_scaled.max().to_numpy(), 1.0)) )
    
        # scaled mins should all be (close to) 0.0
        self.assertTrue( np.all(np.isclose(df_scaled.min().to_numpy(), 0.0)) )
   

class TestWinsorizationOutlierTransformer(unittest.TestCase):
    def test_no_outlier_transformation(self):
        """Transformer supports grid search by allowing for the threshold to be set to 0.0
        A threshold of 0.0 is interpreted as no outlier transformation desired.
        """
        # don't transform outliers in the data
        pipeline = Pipeline([
            ('no_outlier_transformation', WinsorizationOutlierTransformer(outlier_threshold=0.0) ),
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
   
    def test_outlier_transformation(self):
        """Test that the outlier transformer uses Winsorization.  Result should be values
        are clipped to be no further than the indicated std from the mean.
        """
        # try 3.0 standard deviations transformation
        pipeline = Pipeline([
            ('clip_outliers', WinsorizationOutlierTransformer(outlier_threshold=3.0) ),
        ])
        df = get_df_features()
        df_outliers = pipeline.fit_transform(df)

        self.assertFalse(df_outliers is df)
        self.assertFalse(df_outliers.equals(df))
        self.assertEqual(df_outliers.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # standard scaling of data, transforms to mean of 0 and std of 1, can then check if any
        # values are more than 3 standard deviations in magnitude.  However if you
        # standard scaled the clipped values, you can get new outliers, so use standard
        # scaling of original mean and standard deviation
        df_scaled = (df_outliers - df.mean()) / df.std()
        self.assertTrue(df_scaled.min().min() > -3.001)
        self.assertTrue(df_scaled.max().max() < 3.001)
        
        # test clipping at 1.5 std
        pipeline = Pipeline([
            ('clip_outliers', WinsorizationOutlierTransformer(outlier_threshold=1.5) ),
        ])
        df = get_df_features()
        df_outliers = pipeline.fit_transform(df)

        self.assertFalse(df_outliers is df)
        self.assertFalse(df_outliers.equals(df))
        self.assertEqual(df_outliers.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))
        df_scaled = (df_outliers - df.mean()) / df.std()
        self.assertTrue(df_scaled.min().min() > -1.501)
        self.assertTrue(df_scaled.max().max() < 1.501)
        
    def test_outlier_transformation_on_standard_scaled_data(self):
        """Test that the outlier transformer works on already standard scaled data.
        """
        # should work on data that is already standard scaled
        pipeline = Pipeline([
            ('standard_scaling', FeatureScalerTransformer(type_of_scaling='standard')),
            ('clip_outliers',    WinsorizationOutlierTransformer(outlier_threshold=2.5) ),
        ])
        df = get_df_features()
        df_outliers = pipeline.fit_transform(df)
        
        # already standard scaled before outlier clipping, so should already be clipped
        self.assertFalse(df_outliers is df)
        self.assertFalse(df_outliers.equals(df))
        self.assertEqual(df_outliers.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))
        self.assertTrue(df_outliers.min().min() > -2.501)
        self.assertTrue(df_outliers.max().max() < 2.501)
        
    def test_outlier_transformation_on_minmax_scaled_data(self):
        """Test that the outlier transformer works on already minmax scaled data.
        """
        # test outlier removal from min max scaled data
        pipeline = Pipeline([
            ('minmax_scaling', FeatureScalerTransformer(type_of_scaling='minmax')),
        ])
        df = get_df_features()
        df_minmax_scaled = pipeline.fit_transform(df)
        
        pipeline = Pipeline([
            ('clip_outliers',  WinsorizationOutlierTransformer(outlier_threshold=1.25) ),
        ])
        df_outliers = pipeline.fit_transform(df_minmax_scaled)
        
        self.assertFalse(df_outliers is df_minmax_scaled)
        self.assertFalse(df_outliers.equals(df_minmax_scaled))
        self.assertEqual(df_outliers.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))
        
        df_rescaled = (df_outliers - df_minmax_scaled.mean()) / df_minmax_scaled.std()
        self.assertTrue(df_rescaled.min().min() > -1.2501)
        self.assertTrue(df_rescaled.max().max() < 1.2501)
        
        
class TestVIFThresholdTransformer(unittest.TestCase):
    def test_no_vif_clipping(self):
        """Transformer supports grid search by allowing for the threshold to be set to 1.0
        A threshold of 1.0 is interpreted as no vif threshold clipping is desired.
        """
        # don't use vif feature clipping on the data
        pipeline = Pipeline([
            ('no_vif_clipping', VIFThresholdTransformer(feature_ratio=1.0) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))
        
    def test_vif_clip_32_features(self):
        """Test clipping.  We know we want to use 32 values after clipping 30 from
        the vif scores often.  This happens on the cleaned df_features at a ratio of 0.52
        """
        # use vif scores to get features with the lowest 0.52 (52%) of vif scores
        pipeline = Pipeline([
            ('vif_clipping', VIFThresholdTransformer(feature_ratio=0.52) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)
        
        # test number of features changed
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 32))
        
        # the following are the 32 features we git with this current vif procedure.
        # compare to table 1 of reference paper, there is overlap, but they are not
        # exact.  It remains an open issue if there is a problem with vif calculaations
        # here or not.
        best_feature_list = [
            'fixation_duration_median', 
            'fixation_duration_minimum',
            'fixation_duration_skew', 
            'fixation_duration_kurtosis',
            'saccade_duration_median', 
            'saccade_duration_minimum',
            'saccade_duration_skew', 
            'saccade_duration_kurtosis',
            'saccade_amplitude_median', 
            'saccade_amplitude_minimum',
            'saccade_amplitude_skew', 
            'saccade_amplitude_kurtosis',
            'saccade_velocity_sd',
            'saccade_velocity_skew',
            'saccade_velocity_kurtosis',
            'saccade_angle_absolute_median',
            'saccade_angle_absolute_standard_deviation',
            'saccade_angle_absolute_maximum', 
            'saccade_angle_absolute_kurtosis',
            'saccade_angle_relative_median',
            'saccade_angle_relative_standard_deviation',
            'saccade_angle_relative_minimum', 
            'saccade_angle_relative_kurtosis',
            'pupil_diameter_standard_deviation', 
            'pupil_diameter_skew',
            'pupil_diameter_kurtosis',
            'number_of_blinks',
            'blink_duration_mean',
            'number_of_saccades',
            'horizontal_saccade_proportion',
            'fixation_dispersion',
            'fixation_saccade_durtion_ratio']
        best_feature_list_sorted = best_feature_list.copy()
        best_feature_list_sorted.sort()
        
        returned_feature_list_sorted = df_transformed.columns.to_list()
        returned_feature_list_sorted.sort()
        
        self.assertTrue(returned_feature_list_sorted == best_feature_list_sorted)
        
        # The feature values should not have changed for those features selected
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))
        
    def test_vif_clip_num_features(self):
        """Test clipping for various desired number of resulting features.
        """
        # use vif scores to get features with the lowest 0.26 (26%) of vif scores,
        # should result in 16 features this time.
        pipeline = Pipeline([
            ('vif_clipping', VIFThresholdTransformer(feature_ratio=0.26) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 16))
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))

        # use vif scores to get features with the lowest 0.10 (10%) of vif scores,
        # should result in 6 features this time.
        pipeline = Pipeline([
            ('vif_clipping', VIFThresholdTransformer(feature_ratio=0.10) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 6))
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))

        
class TestFeatureSelectionTransformer(unittest.TestCase):
    def test_no_feature_selection(self):
        """Transformer supports grid search by allowing for the type of selection to be
        set to none, or for the feature ratio to be 1.0.  In either case, no feature 
        selection should be performed.
        """
        # don't use feature selection on the data
        mind_wandered_label = get_mind_wandered_label()
        participant_ids = get_participant_ids()
        
        pipeline = Pipeline([
            ('no_feature_selection', FeatureSelectionTransformer(mind_wandered_label, participant_ids, feature_ratio=1.0) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # don't use feature selection on the data when use 'none' for the type of feature selection
        pipeline = Pipeline([
            ('no_feature_selection', FeatureSelectionTransformer(mind_wandered_label, participant_ids, type_of_selection='none') )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # don't use feature selection when the type of selection is none, no matter what the ratio is set as
        pipeline = Pipeline([
            ('no_feature_selection', FeatureSelectionTransformer(mind_wandered_label, participant_ids, type_of_selection='none', feature_ratio=0.5) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

        # don't use feature selection when the ratio is 1.0 no matter what the type of selection says
        pipeline = Pipeline([
            ('no_feature_selection', FeatureSelectionTransformer(mind_wandered_label, participant_ids, type_of_selection='correlation-selection', feature_ratio=1.0) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertTrue(df is df_transformed)
        
        # test data didn't actually change
        df_original = get_df_features()
        self.assertFalse(df_transformed is df_original)
        self.assertTrue(df_transformed.equals(df_original))
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, CLEAN_NUM_FEATURES))

    def test_correlation_feature_32_selection(self):
        """Transformer test that we get 32 features whne using appropriate ratio.
        """
        # needed for the feature selector
        mind_wandered_label = get_mind_wandered_label()
        participant_ids = get_participant_ids()
        
        # select features using correlation scores.
        pipeline = Pipeline([
            ('correlation_feature_selection', 
                 FeatureSelectionTransformer(mind_wandered_label, participant_ids, 
                                             type_of_selection='correlation-selection', feature_ratio=0.52) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)

        # test number of features changed
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 32))
        
        # the following are the 32 features we git with this current correlction
        # feature selection.  Like the vif procedure, this could use some more scrutinity
        # though when compared to a tree based feature selection it looked pretty close.
        # But I might just drop this and use stanard scikit-learn feature selection methods.
        best_feature_list = [
            'pupil_diameter_median',
            'pupil_diameter_mean',
            'saccade_amplitude_range',
            'saccade_amplitude_maximum',
            'fixation_duration_maximum',
            'fixation_duration_range',
            'fixation_duration_standard_deviation',
            'pupil_diameter_maximum',
            'pupil_diameter_minimum',
            'saccade_amplitude_standard_deviation',
            'saccade_angle_relative_mean',
            'saccade_angle_relative_skew',
            'saccade_duration_maximum',
            'saccade_duration_range',
            'saccade_duration_standard_deviation',
            'saccade_angle_absolute_mean',
            'saccade_angle_absolute_skew',
            'saccade_velocity_mean',
            'saccade_velocity_median',
            'saccade_amplitude_skew',
            'saccade_duration_kurtosis',
            'saccade_duration_skew',
            'saccade_amplitude_kurtosis',
            'saccade_duration_mean',
            'saccade_velocity_range',
            'saccade_angle_absolute_median',
            'saccade_velocity_sd',
            'pupil_diameter_range',
            'pupil_diameter_standard_deviation',
            'fixation_duration_skew',
            'fixation_duration_mean',
            'saccade_angle_relative_median'
        ]

        # actually this method uses a random sample to determine
        # the best features, so it is not necessarily stable (can get
        # different list of best features on different calls for same
        # data).  This is another reason it is maybe not as good a choice?
        #best_feature_list_sorted = best_feature_list.copy()
        #best_feature_list_sorted.sort()
        #returned_feature_list_sorted = df_transformed.columns.to_list()
        #returned_feature_list_sorted.sort()
        #self.assertTrue(returned_feature_list_sorted == best_feature_list_sorted)
        
        # The feature values should not have changed for those features selected
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))

    def test_correlation_feature_num_selection(self):
        """Test correlation based feature selection for various desired 
        number of resulting features.
        """
        # needed for the Feature selector
        mind_wandered_label = get_mind_wandered_label()
        participant_ids = get_participant_ids()
        
        # use correlation feature selection to get indicated ratio of features,
        # should result in 16 features this time.
        pipeline = Pipeline([
            ('correlation_feature_selection', 
                 FeatureSelectionTransformer(mind_wandered_label, participant_ids, 
                                             type_of_selection='correlation-selection', feature_ratio=0.26) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 16))
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))

        # use correlation feature selection to get best 10% (0.10 ratio) of features,
        # should result in 6 features this time.
        pipeline = Pipeline([
            ('correlation_feature_selection', 
                 FeatureSelectionTransformer(mind_wandered_label, participant_ids, 
                                             type_of_selection='correlation-selection', feature_ratio=0.10) )
        ])
        df = get_df_features()
        df_transformed = pipeline.fit_transform(df)
        self.assertFalse(df is df_transformed)
        self.assertEqual(df_transformed.shape, (CLEAN_NUM_TRIALS, 6))
        returned_feature_list = df_transformed.columns
        df_original = df[returned_feature_list]
        self.assertTrue(df_transformed.equals(df_original))
        
        
        
if __name__ == '__main__':
    unittest.main()
