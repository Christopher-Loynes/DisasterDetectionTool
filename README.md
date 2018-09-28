# An automated disaster detection tool written in Python as part of an MSc thesis.

A text classifier is used to classify tweets into disaster-type. These classified tweets are then clustered using *Density-Based Spatial Clustering of Applications with Noise (DBSCAN)*. To select the most relevant text classifier and parameter settings, 2 stages of assessment are completed: *testing* and *evaluation*.

## Testing 
  - 4 text classifiers are tested. In the thesis, this involved 8 different parameter settings, each parameter unique to the classifer
    - *Naive Bayes (benchmark)*
    - *Linear Support Vector Classifier (SVC)*
    - *Logisitc Regression*
    - *Random Forest*
    
  - All classifiers tested using:
    - *Ten-fold cross-validation*
    - *Stratified k-fold sampling*
    - *Micro-average AUC scores (computed using ROC curves)*
    - *Micro-average Accuracy scores*
    
  - An optimised Naive Bayes classifier and the best-performing classifier (found to the linear SVC) are then selected for *evaluation*
  
## Evaluation 
  - The optimised Naive Bayes classifier and best-performing classifier are assessed further:
    - *SMOTE oversampling*
    - *Macro-average AUC scores*
    - *Macro-average Accuracy scores*
    - *Confusion matrices*
    - *Review of misclassified tweets*
    
  - *DBSCAN* algorithm is evaluated using tweets classified by both text classifiers 

After *testing* and *evaluation*, the MSc thesis proposed the use of a linear SVC with a 'cost' parameter of 0.5 and a DBSCAN algorithm with an 'eps' value of 20 (20km) and 'min_samples' value of 80 (80 tweets). This means a disaster is detected if 80 tweets, classified by the linear SVC as belonging to the same disaster-type (e.g. typhoon), are clustered inside a radius of 20km, using coordinates appended to tweets.

# Resources

## The ‘Data’ folder contains 2 files:

**_1) Agg_Reduced.csv.zip_** - a reduced corpus of raw tweets relating to man-made and natural disasters downloaded from http://crisisnlp.qcri.org/

**_2) gaz.csv.zip_** - an offline gazetteer, downloaded from http://download.geonames.org/export/dump/



## The ‘Preprocessing’ folder contains 2 scripts:

**_1) Aggregate_Raw_Tweet_Files_** - aggregate tweets for individual disasters downloaded from http://crisisnlp.qcri.org/

**_2) Normalise_Tweets_** - text normalisation of all aggregated raw tweets



## The ‘Testing’ folder contains 3 scripts:

**_1) Unigrams_** - test the 4 text classifiers on unigrams using Term Frequency Inverse Document Frequency (TF-IDF), ten-fold cross validation and stratified K-sampling. Different tests can be performed for each classifier, adjusting the values of classifier-specific parameters (lines 607-610). The micro-average Accuracy and AUC values (using ROC curves) are calculated, since the 'Agg_Reduced.csv.zip' dataset contains a class imbalance.

**_2) Bigrams_** - test the same text classifiers via the same techniques but on bigrams. The code is identical, except the 'ngram_range' in line 583, as part of the TfidfVectorizer and the subsequent use of the TF-IDF vector.

**_3) 'Trigrams_** - test the same text classifiers via the same techniques but on trigrams. The code is identical, except the 'ngram_range' in line 583, as part of the TfidfVectorizer and the subsqeuent use of the TF-IDF vector.


## The ‘Evaluation’ folder contains 1 script:

**_Overall_Evaluation_** - the best variant of the Naive Bayes (benchmark) and linear Support Vector Classifier (the best performing classifier from 'testing' performed for the MSc thesis) are evaluated on increasingly larger subsets. This is controlled by adjusting the `p` value in line 17. Unigrams is used, based on the results obtained during the thesis but can be changed by adjusting the `ngram_range` in line 588. 

SMOTE oversampling is performed on the training dataset only, not the test dataset, to prevent artifically creating an event, since the volume of tweets in a specified radius is used for DBSCAN clustering. No cross-validation is performed. 

Confusion matrices are created, ROC curves and macro-average AUC values are calculated, alongside macro-average Accuracy values (since the SMOTE oversampling removes the class imbalance), and misclassified tweets reviewed. 

DBSCAN clustering is performed on classified tweets from both the Naives Bayes and SVC classifiers. The `epsilon value` (radius) in line 1002 can be adjusted (this value is for both Naive Bayes and SVC) and the `min_density` (number of points in the specified radius) in line 1029 for the SVC and line 1280 for the Naive Bayes, can be adjusted. 

A colour-coded map of the world is generated for each classifier that overlays the location and event-type of each detected disaster, alongside a .csv export, which contains the location, event-type and coordinates of each disaster.

