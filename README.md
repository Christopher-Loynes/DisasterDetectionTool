An automated disaster detection tool written in Python

The Disaster Detection tool was built as part of an MSc thesis.

The ‘Data’ folder contains two files:

’Agg_Reduced.csv.zip’ - a reduced corpus of raw tweets relating to man-made and natural disasters downloaded from
http://crisisnlp.qcri.org/

’gaz.csv.zip’ - an offline gazetteer, downloaded from http://download.geonames.org/export/dump/

The ‘Preprocessing’ folder contains two scripts:

’Aggregate_Raw_Tweet_Files’ - aggregate tweets for individual events downloaded from http://crisisnlp.qcri.org/

’Normalise_Tweets’ - text normalisation of all aggregated tweets

The ‘Testing’ folder contains 3 scripts:

‘Unigrams’ - test a Naive Bayes (benchmark), linear Support Vector Classifier, Logistic Regression and Random Forest text classifier
on unigrams using Term Frequency Inverse Document Frequency (TF-IDF), ten-fold cross validation and stratified K-sampling. Different 
tests can be performed for each classifier, adjusting the valyes of classifier-specific parameters. The micro-average Accuracy and 
AUC values (using ROC curves) are calculated, since the 'Agg_Reduced.csv.zip' dataset contains a class imbalance.

‘Bigrams’ - test the same text classifiers via the same techniques but on bigrams

‘Trigrams’ - test the same text classifiers via the same techniques but on trigrams

The ‘Evaluation’ folder contains 1 script:

Overall_Evaluation - the best variant of the Naive Bayes (benchmark) and highest performing classifier (linear Support Vector Classifier)
are evaluated on increasingly larger subsets. This is controlled by adjusting the p value in line 17. Misclassified tweets are reviewed,
ROC curves and macro-average AUC values are calculated, alongside macro-average Accuracy values. DBSCAN clustering is performed on 
classified tweets from both the Naives Bayes and SVC classifiers. A colour-coded map of the world is generated that overlays the 
location and event-type of each detected disaster, alongside a .csv export, which contains the location, event-type and coordinates 
of each disaster.
