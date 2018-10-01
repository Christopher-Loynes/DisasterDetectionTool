# An automated disaster detection tool written in Python as part of an MSc thesis

A text classifier is used to classify tweets into disaster-type. These classified tweets are then clustered using *Density-Based Spatial Clustering of Applications with Noise (DBSCAN)*. A disaster is detected if the number of tweets belonging to a single disaster-type exceeds a threshold within a specified radius. In the thesis, the values proposed are 80 tweets and 20km, respectively.

Process completed in thesis:

1) Pre-process raw tweets
2) 'Testing'
    - 4 text classifiers tested
3) 'Evaluation'
    - Best version of the benchmark and best performing text classifier evaluated further
    - DBSCAN algorithm evaluated on tweets classified by both of the text classifiers  
4) Identificaton of best text classifier and parameter values, alongside the best parameter values for the DBSCAN algorithm

## Example of Colour-Coded Colour-Coded Export

![nbclusterplot100](https://user-images.githubusercontent.com/34406492/46284225-eb539d00-c56e-11e8-9689-50f34a9a26d8.png)


## Resources

- [1) 'Data' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Data'-Folder)
- [2) 'Pre-processing' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Preprocessing'-Folder)
- [3) 'Testing' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Testing'-Folder) 
- [4) 'Evaluation' folder](https://github.com/Christopher-Loynes/DisasterDetectionTool/wiki/'Evaluation'-Folder)



