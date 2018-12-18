# An automated disaster detection tool written in Python as part of an MSc thesis

# Summary

Developed as part of my MSc dissertation at the University of Edinburgh (2017/18), titled "The Detection and Location Estimation of Disasters Using Twitter and the Identification of Non-Governmental Organisations (NGOs) Using Crowdsourcing".

A text classifier is used to classify tweets into disaster-types. These classified tweets are then clustered using *Density-Based Spatial Clustering of Applications with Noise (DBSCAN)*. A disaster is detected if the number of tweets belonging to a single disaster-type exceeds a threshold within a specified radius. In the thesis, the initial values proposed are 80 tweets and 20km, respectively. These should be adjusted based on the volume and velocity of tweets being processed.

# Process

1) [**Pre-processing**](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Preprocessing)
    - Aggregate raw tweets
    - Text normalisation
        - Remove "RT", which denotes 'retweet'
        - Remove hyperlinks
        - Remove hash signs (#) from hashtags, so only the characters/word remains
        - Tokenise tweets
        - Remove excess characters repeated 3 or more times
        - Remove username handles
        - Remove any 'pic.twitter' URLs
        - Remove any URL
        - Expand contractions into full words
        - Remove stop words from tweet
        - Remove punctuation
        - Remove numerical characters
        - Remove any tweets remaining that contain less than 4 words
2) [**Testing**](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Testing)
    - 4 text classifiers tested:
        - Logistic Regression
        - Naive Bayes (benchmark)
        - Random Forest
        - Support Vector Classifier
3) [**Evaluation**](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Evaluation)
    - Best version of the benchmark and the best performing text classifier evaluated further
    - DBSCAN algorithm evaluated on tweets classified by both of the text classifiers 
    - Identificaton of best text classifier and parameter values
    - Identification of best parameter values for the DBSCAN algorithm

## Exports

**All disasters detected are exported in 2 formats.**

1) The geographic location of events are overlayed on a map. Each colour indicates a different disaster-type.
    - Example below is based on tweets classified using a Naive Bayes text classifier

![nbclusterplot100](https://user-images.githubusercontent.com/34406492/46284225-eb539d00-c56e-11e8-9689-50f34a9a26d8.png)

2) .CSV export that contains:
     - Disaster type
     - Name of the geographic location
     - Coordinates of the geographic location

## Resources

- [1) Data](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Data)
- [2) Pre-processing](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Preprocessing)
- [3) Testing](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Testing) 
- [4) Evaluation](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/Evaluation)



