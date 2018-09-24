# DisasterDetectionTool
An automated disaster detection tool written in Python


The Disaster Detection tool was built as part of an MSc thesis. 


The ‘Data’ folder contains two files:

1) ’Agg_Reduced.csv.zip’ - a reduced corpus of raw tweets downloaded from http://crisisnlp.qcri.org/

2) ’gaz.csv.zip’ - a gazetteer, provided by Genomes and downloaded from http://download.geonames.org/export/dump/



The ‘Preprocessing’ folder contains two scripts:

1) ’Aggregate_Raw_Tweet_Files’ - aggregate tweets for individual events downloaded from http://crisisnlp.qcri.org/

2) ’Normalise_Tweets’ - text normalisation of all aggregated tweets



The ‘Testing’ folder contains 3 scripts:

1) ‘Unigrams’ - test a Naive Bayes (benchmark), linear Support Vector Classifier, Logistic Regression and Random Forest 
text classifier on unigrams using Term Frequency Inverse Document Frequency (TF-IDF), ten-fold cross validation and stratified 
K-sampling. Different tests can be performed for each classifier, adjusting the valyes of classifier-specific parameters. 

2) ‘Bigrams’ - test the same text classifiers via the same techniques but on bigrams

3) ‘Trigrams’ - test the same text classifiers via the same techniques but on trigrams



The ‘Evaluation’ folder contains 1 script:

1) Overall_Evaluation - the best variant of the Naive Bayes (benchmark) and highest performing classifier (linear Support Vector
Classifier) are evaluated on increasingly larger subsets. This is controlled by adjusting the p value in line 17. Misclassified 
tweets are reviewed, ROC curves and AUC values are calculated, alongside DBSCAN clustering. A colour-coded map is generated map 
of the world is generated that overlays the location and event-type of each detected disaster, alongside an .csv export of the 
location, event-type and coordinates.
