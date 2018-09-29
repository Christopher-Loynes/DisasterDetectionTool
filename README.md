# An automated disaster detection tool written in Python as part of an MSc thesis

A text classifier is used to classify tweets into disaster-type. These classified tweets are then clustered using *Density-Based Spatial Clustering of Applications with Noise (DBSCAN)*. A disaster is detected if the number of tweets belonging to a single disaster-type exceeds a threshold within a specified radius. In the thesis, the values proposed are 80 tweets and 20km, respectively.

After pre-processing raw tweets, the most relevant text classifier and parameter values are selected using 2 stages of assessment: *testing* and *evaluation*. Both are explained below.

## Testing 
  - 4 text classifiers are tested. In the thesis, this involved 8 different values for each parameter. Each parameter ia unique to the classifer. Below is the name of each classifier tested and the parameter adjusted.
    - *Naive Bayes (benchmark)*: `alpha`
    - *Linear Support Vector Classifier (SVC)*: `cost`
    - *Logistic Regression*: `cost`
    - *Random Forest*: `number of trees`, `splitting criterion`, `maximum tree depth`
    
  - All text classifiers tested using:
    - *Term Frequency Inverse Document Frequency (TF-IDF)*
      - *Unigrams, Bigrams & Trigrams*
    - *Ten-fold cross-validation*
    - *Stratified K-fold sampling*
    - *Micro-average AUC scores (computed using ROC curves)*
    - *Micro-average Accuracy scores*
    
  
## Evaluation 
  - The optimised *Naive Bayes* classifier and best-performing classifier (found to be the *linear SVC*) are assessed further:
    - *Term Frequency Inverse Document Frequency (TF-IDF)*
    - *SMOTE oversampling*
      - *Training data only*
    - *Macro-average AUC scores*
    - *Macro-average Accuracy scores*
    - *Confusion matrices*
    - *Review of misclassified tweets*
    
  - *DBSCAN* algorithm is evaluated using tweets classified by both text classifiers 
    - Performance is evaluated after adjusting `eps` and `min_samples` parameter values
    -  `eps`: radius (input is KM)
    - `min_samples`: minimum number of tweets in specified radius (`eps`) to detect a disaster

After *testing* and *evaluation*, the MSc thesis proposed the use of a *linear SVC* with a `cost` parameter of 0.5 and a *DBSCAN* algorithm with an `eps` value of 20 (20km) and `min_samples` value of 80 (80 tweets). This means a disaster is detected if 80 tweets, classified by the *linear SVC* as belonging to the same disaster-type (e.g. typhoon), are clustered inside a radius of 20km, using coordinates appended to tweets.

## Resources

- [1) 'Data' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Data'-Folder)
- [2) 'Pre-processing folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Preprocessing'-Folder)
- [3) 'Testing' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Testing'-Folder)
- [3) ['Evaluation' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Evaluation'-Folder)
