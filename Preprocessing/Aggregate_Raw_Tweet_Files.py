 # import libraries
import pandas as pd
import glob

############################################################################################
#################### MERGE ALL FILES THAT CONTAIN PROCESSED TWEETS #########################
############################################################################################
        
path = r'/Users/christopherloynes/Desktop/Tweets'
filenames = glob.glob(path + "/*_preprocessed.csv")

# concatenate all individual .csv files that were created in the previous pre-processing stage
agg = pd.concat([pd.read_csv(f) for f in filenames])

# combined dataset is exported to .csv for archiving purposes
agg.to_csv("agg.csv", index=False)

agg = pd.read_csv("agg.csv")