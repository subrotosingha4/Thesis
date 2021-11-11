# Tasks to Complete

1. Replicate figure 2 histogram of auc-roc with our own grid search of models.  Suggest we try the following models
   - Logistic Regression
   - k-nn
   - Bayesian classifier
   - simple decision tree
   - Support Vector Machine (SVM)
   - Random Forest
   
   This isn't the exact same list of classifers tried in the original paper.  But our replication goal is to see if
   a similar histogram of auc-roc scores is achieved, and also important, that we can match the best reported
   model (reported as  LogisticRegression model with an auc-roc score of 0.64)
   
   We should try the following grid search options for the different model types
   - Use vif selection, don't use (I'm not sure if the vif selection currently in this project is working?)
   - Feature select.  Maybe we can try both their reported method based on similarity scores, and a more
     standard tree based method.  Might want to try more levels, say 10%, 20%, ... 100% of features, and maybe
     try including all features (they always selected, but if we go to 100% then we can try without selecting
     features).
   - Use standard scaling vs. don't use any scaling
   - Trim outliers, vs. don't.  Could also try to grid the threshold, like 2, 3, 4 standard deviaitons for the trim.
   - The may want to try some specific model parameter grid points, maybe at least vary amount of regularization a bit
     on LR, SVM, and tree depth on tree classifiers
     
2. If get models as good as in paper, report it and make figures of confusion matrix and auc-roc curve for it.
   - Might want to look at worst model as well?  Actually we have a question about their figure 2.  They show
     a model with a 0.27 auc-roc score.  When your score is low, you can essentially create a model that just guesses
     the opposite of what that model guesses (if it is a binary classification task), and in this case it would imply
     that whatever that model was doing, you could get a 1.0 - 0.27 = 0.73 auc-roc, which is much better than they
     reported.
     
3. If we can replicate, the next step is to try and create a model that performs better.  If with our grid search we find
   a classic ML classifier that performs better than 0.64 auc-roc, then great.  If not, we should first start with simple
   feed-forward neural networks.
   - grid search as above, and on the model architecture a bit (number of hiddne layers, size of hidden layers, regularization,
     maybe dropout layers).
     
4. After that, if we can, should try some true deep learning networks.  Not sure if an argument can be made that convolutions would
   be useful here, but worth trying.  Could try 1-d convolutions.  In either case might want to try to rearrnange the grouping
   of features that are input carefully, perhaps this should be a metaparameter (since convolutions try to find local features,
   thus the question is if there are any useful local patterns in the summary eye tracking statistics here).
   
5. We can also try recurrent networks.  At this point I am not sure if RNN will help here.  We have 135 participants, with
   number of trials ranging from all 57 down to 4.  Maybe we could try and build trial sequences.  For example, if
   participant has all 57 trials, we could use a sequence of 4 to feed in in groups, e.g. trial 1-2-3-4, trial 2-3-4-5,
   trial 3-4-5-6, etc.  Not sure if should overlap, or try 1-2-3-4, 5-6-7-8.  Also a lot of subjects have a lot of
   missing trials in the sequence, maybe we shouldn't stitch together over a hole?  For example, if we have trials 1-10 but
   5 is missing, and trying sequences of size 4, should we only do sequences 1-2-3-4, 6-7-8-9, 7-8-9-10?
   The `segment_index` in the experiment metadata essentially identifies this trial sequence.  
   Another issue is curse of dimensionality.  If we feed in all 62 features, even for a window of size 3 or 4, that
   is 180-240 features.  May want to feature select based on results of feature selection previously.
   
   And really need to brush up on RNN.  Should we really just be feeding them in 1 at a time, but in correct
   chronological sequence (e.g. segment_index 1, 2, 3, ...), which some indication of when the trial ends after the last
   sequence, or when we come to a whole in the sequence?


# Tables and Figures to Produce

- Produce a table of the 62 features used, a type of data dictionary.  Probably group into the 4 groups mentioned.
- Our own auc-roc histogram results
- Would a table of features ranked by the correlation score and by a tree method be useful?
- Likewise a table of feature importance to any good models (like reference paper table 1) might be good, especially for best model
  we can find if it is beating the results reported in this paper.
- For best model in each task above, get confusion matrix and auc-roc curve.
   