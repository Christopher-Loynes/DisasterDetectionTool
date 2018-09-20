 # import libraries
import pandas as pd
import glob
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer

############################################################################################
################################### PRE-PROCESS TWEETS #####################################
############################################################################################

# import all raw tweets that are based in several .csv files with the ending '_raw.csv'
path = r'/Users/christopherloynes/Desktop/Tweets'
filenames = glob.glob(path + "/*_raw.csv")

# for every .csv file (each file is an event)
for a in filenames:
    # import the values in the .csv file
    data = pd.read_csv(a)
    
    # extract raw tweet, co-ordinates & event-type
    message = data['text']
    
    # extract coordinates and event_type. turn both into dataframes
    coordinates = data['coordinates']
    coordinates = pd.DataFrame(coordinates)
    
    event_type = data['event_type']
    event_type = pd.DataFrame(event_type)

    # create a list, which will be populated with each tweet after it is pre-processed
    preprocessed = []
    
    # loop through each tweet in the .csv file
    for b in range(len(message)):
        
        # select row
        tweet = message.iloc[b]
        
        # remove "RT", which denotes 'retweet'
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
        # remove hash signs (#) from hashtags, so only the characters/word remains
        tweet = re.sub(r'#', '', tweet)
    
        # tokenise tweets
        # case folding using 'preserve case' = false
        # remove excess characters repeated 3 or more times (e.g. yeeeah), using 'reduce_len'
        # remove username handles using the 'strip_handles' feature
        tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
        filtered_words = tokenizer.tokenize(tweet)
        
        # Remove any 'pic.twitter' URLs
        prefixes = ('pic.','.twitter.')
        for word in filtered_words:
            if word.startswith(prefixes):
                filtered_words.remove(word)
        
        # Remove any URL, even if hidden in (), {} etc.
        for obje in filtered_words:
            if obje.startswith('http'):
                filtered_words.remove(obje)
   
        # Expand contractions into full words, e.g. 'ain't' expanded to 'are not'
        contractions = { 
        "ain't": "are not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'd've": "he would have",
        "he'll": "he shall / he will",
        "he'll've": "he shall have / he will have",
        "he's": "he has / he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I had / I would",
        "I'd've": "I would have",
        "I'll": "I shall / I will",
        "I'll've": "I shall have / I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'd've": "it would have",
        "it'll": "it shall / it will",
        "it'll've": "it shall have / it will have",
        "it's": "it has / it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had / she would",
        "she'd've": "she would have",
        "she'll": "she shall / she will",
        "she'll've": "she shall have / she will have",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as / so is",
        "that'd": "that would / that had",
        "that'd've": "that would have",
        "that's": "that has / that is",
        "there'd": "there had / there would",
        "there'd've": "there would have",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'd've": "they would have",
        "they'll": "they shall / they will",
        "they'll've": "they shall have / they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what'll've": "what shall have / what will have",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "when's": "when has / when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has / where is",
        "where've": "where have",
        "who'll": "who shall / who will",
        "who'll've": "who shall have / who will have",
        "who's": "who has / who is",
        "who've": "who have",
        "why's": "why has / why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had / you would",
        "you'd've": "you would have",
        "you'll": "you shall / you will",
        "you'll've": "you shall have / you will have",
        "you're": "you are",
        "you've": "you have"
        }  
        
        for item in filtered_words:
            if "'" in item:
                try:
                    fullitem = contractions[item]
                except:
                    fullitem = item
                    pass
                filtered_words[filtered_words.index(item)] = fullitem
         
        #Remove stop words from tweet and join the filtered tokens into a string
        tweet = ' '.join(filtered_words)
        
        # Remove any remaining URLs
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", tweet)
        
        # Remove punctuation
        punctokenizer = RegexpTokenizer(r'\w+')
        tweet = punctokenizer.tokenize(tweet)
        
        # Remove numerical characters from string
        result = [l for l in tweet if not l.isdigit()]
        numwords = len(result)
        
        # join together to form a string
        result = ' '.join(result)
        
        # If results < 4 words, the tweet is not used. prevents needless processing of 
        # tweets with no value
        if numwords < 4:
            result = 0
        preprocessed.append(result)
        
    # Add 'preprocessed' and 'event_type' to dataset
    preprocessed = pd.DataFrame(preprocessed, columns=['preprocessed_text'])
    message = pd.concat([preprocessed,event_type,coordinates], axis=1)
    
    # Remove any tweets with less than 4 words
    message = message[message.preprocessed_text != 0]
    
    # export the processed tweets in a .csv file
    newname = a + "_preprocessed.csv"
    message.to_csv(newname, index=False)
    

