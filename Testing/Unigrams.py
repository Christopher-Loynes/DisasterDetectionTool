# import libraries
from geotext import GeoText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import Nominatim
import random

print("start importing raw tweets")

filename = "agg_reduced.csv"
p = 0.3333  # % of the lines
# keep the header, then take only % of lines
# if random from [0,1] interval is > than %, the row will be skipped
agg = pd.read_csv(filename, header=0, skiprows=lambda i: i>0 and random.random() > p)

print("imported raw tweets")

############################################################################################
############ FIND LOCATIONS STATED IN A PROCESSED TWEET THAT IS NOT GEOTAGGED ##############
############################################################################################

# store number of tweets imported (aggregate)
num_tweets_imported = len(agg)

# create new dataframe for processed tweets with no coordinates
agg_nocord = agg[agg.coordinates == "NA,NA"]

# store the number of tweets that are not geotagged
num_non_geotagged_tweets = len(agg_nocord)

# create a new dataframe for processed tweets with coordinates
agg_cord = agg[agg.coordinates != "NA,NA"]
agg_cord = agg_cord.reset_index()
agg_cord = agg_cord.drop(['index'], axis=1)

# store number of tweets that are geotagged 
num_geotagged_tweets = len(agg_cord)

# separate the coordinate column in the 'agg_cord' dataframe to get lat & long values
agg_split = agg_cord['coordinates'].str.split(',', expand=True)
agg_split.columns = ['lat','long']

# Now we have lat & long values, concatenate with agg_cord (before the coordinates were split)
# and drop unwanted column (coordinates) and rename column headings
agg_cord = pd.concat([agg_cord, agg_split], axis=1)
agg_cord = agg_cord.drop(['coordinates'], axis=1)
agg_cord.columns = ['preprocessed','event_type','lat','long']

# extract 'preprocessed' (processed tweets) from 'agg_nocord' (tweets not geotagged) &
# turn from series into list
preprocessed = agg_nocord['preprocessed_text']
preprocessed = pd.DataFrame(preprocessed)
preprocessed = preprocessed['preprocessed_text'].values.tolist()

# extract 'event_type' from 'agg_nocord' (tweets not geotagged) and turn into a dataframe
event_type = agg_nocord['event_type']
event_type = pd.DataFrame(event_type)

# reset the index in the dataframe 'event_type' and drop the additional column created
event_type = event_type.reset_index()
event_type = event_type.drop(['index'], axis=1)

# create lists to append the names of countries and cities contained in a processed tweet
# that is not geotagged
countries = []
cities = []

print("find locations in tweets")

# extract geographic locations (countries & cities) from the content of tweets not geotagged
# loop through 'preprocessed', which was taken from 'agg_nocord' (tweets not geotagged)
for tweet in preprocessed:
    try:
        # input a tweet into the function 'GeoText', which identifies geographic locations
        # contained within text
        places = GeoText(tweet)
        # append countries identified in the list 'countries'
        countries.append(str(places.countries[0]))
        # append cities identified in the list 'cities'
        cities.append(str(places.cities))
    except:
        # if no geographic location is found by the function, populate the 2 lists with 'N/A'
        countries.append("N/A")
        cities.append("N/A")
        
print("found locations in tweets")
   
# turn lists into dataframes        
cities = pd.DataFrame(cities, columns=['city'])
countries =  pd.DataFrame(countries, columns=['country'])
preprocessed =  pd.DataFrame(preprocessed, columns=['preprocessed'])

# join 'cities', 'countries', 'preprocessed' & 'event_type' dataframes together
city_country = pd.concat([preprocessed,cities,countries,event_type], axis=1)

# store the number of tweets where no countries or cities were found in the body of a tweet
num_tweets_nocity_nocountry = len(city_country[(city_country.country == "N/A") &
                                               (city_country.city == "N/A")])

# store the number of tweets containing a city, a country or both in the body of a tweet
num_tweets_city_or_country_or_both = len(city_country[(city_country.country != "N/A") & 
                                             (city_country.city == "N/A")]) + len(city_country[(city_country.country == "N/A") &
                                             (city_country.city != "N/A")]) + len(city_country[(city_country.country != "N/A") & 
                                             (city_country.city != "N/A")])

# drop all entries where the country and city locations are 'N/A' (no locations found at all)
city_country = city_country[(city_country.country != "N/A") & (city_country.city != "N/A")]

# reset the index in 'city_country' and drop the new column created   
city_country = city_country.reset_index()
city_country = city_country.drop(['index'], axis=1)

# extract cities from 'city_country' so that the single speech marks and square brackets 
# produced by the function 'GeoText' (identifies locations in tweets) can be removed
cities = city_country['city']
cities = pd.DataFrame(cities)
cities = cities['city'].values.tolist()

# create a new list called 'cities_n' to remove unwanted characters from the GeoText output
# this is not a problem for countries, just cities
cities_n = []

# remove unwanted characters from the GeoText output for cities (no issue for countries)
for a in cities:
    # remove square brackets
    new = str(a).replace('[','').replace(']','')
    # remove single speech marks
    new = str(new).replace('\'','').replace('\'','')
    # append the newly amended city names into the list 'cities_n'
    cities_n.append(new)
    
### if 2 cities are found in a tweet by Geotext, a comma exists. Only take the first location ###

# populate the new cities in a new list called 'cities_new' (the 1st if there are 2)
cities_new = []

# loop through all cities, split where the delimiter ',' appears and take the first element 
for text in cities_n:    
    rest = text.split(',', 1)[0]
    cities_new.append(rest)

# turn the 'cities_new' list into a dataframe
cities_new = pd.DataFrame(cities_new, columns=['city'])  

# drop the 'city' column from 'city_country', will be replaced by the amended city names
city_country = city_country.drop(['city'], axis=1)

# add 'cities_new' to the dataframe 'city_country'
city_country = pd.concat([city_country, cities_new], axis=1)

# replace blank cells with 'N/A' in 'city_country'
city_country.replace(r'^\s*$', "N/A", regex=True, inplace=True)

# replace 'nan' values in the 'city' column in the with 'N/A' for consistency
city_country['city'] = city_country['city'].fillna('N/A')

############################################################################################
######## CREATE LISTS TO BE ITERATED OVER AND COORDINATES OBTAINED FROM GAZETTEERS #########
############################################################################################

# turn 'cities_new' into list, taken from the dataframe 'city_country'
# this is taken from 'city_country' now cells that were blank or NaN have been dealt with
cities_new = city_country['city']
cities_new = pd.DataFrame(cities_new)
cities_new = cities_new['city'].values.tolist()

# turn 'countries' into a list, taken from the dataframe 'city_country'
# this is taken from 'city_country' now cells that were blank or NaN have been dealt with
countries = city_country['country']
countries = pd.DataFrame(countries)
countries = countries['country'].values.tolist()

# make all countries & cities capitalised, to prevent duplicate countries with case variation
# e.g. "Pakistan", "pakistan" & 'PAKISTAN' all become 'Pakistan'
countries = [element.title() for element in countries] 
cities_new = [element.title() for element in cities_new] 

# store the number of tweets where a country and city was found by Geotext in the body
num_tweets_country_and_city = len(city_country[(city_country.country != "N/A") & (city_country.city != "N/A")])

# store the number of tweets where a city was found but not a country in the body of the tweet
num_tweets_country_no_city = len(city_country[(city_country.country != "N/A") & (city_country.city == "N/A")])

# store the number of tweets where a city was found but not a country in the body of the tweet
num_tweets_city_no_country = len(city_country[(city_country.country == "N/A") & (city_country.city != "N/A")])

############################################################################################
############################# LOAD THE OFFLINE GAZETTEER ###################################
############################################################################################

# since there is a limit on the number of times you can call an online gazetteer, an offline
# version is imported with 7.2 million entries. this is called before using an online equivalent

print("import gazetteer")

# load an offline gazetteer for offline geocoding 
gaz = pd.read_csv('gaz.csv')

print("gazetteer imported")

############################################################################################
### CREATE SETS OF 'COUNTRIES' AND 'CITIES' lISTS FOR QUICKER COORDINATE IDENTIFICATION ####
############################################################################################

# create a set of 'cities_new', which removes all duplicates,. prevents repeatedly performing a 
# a 'vlookup' of the same city to obtain its coordinates in the offline gazetteer
cities2 = set(cities_new)

# create a set of 'countries', which removes all duplicates. prevents repeatedly performing a 
# a 'vlookup' of the same city to obtain its coordinates in the offline gazetteer
countries2 = set(countries)

############################################################################################
######## CREATE DICTIONARIES TO POPULATE COORDINATES AND RUN REQUIRED FUNTIONS #############
############################################################################################

# create dictionaries to store the coordinates of locations found in the gazetteer
# this will be searched before the gazetteer for each tweet for optimisation
city_lat_dic = {}
city_long_dic = {}

# online function from the library 'geolocator'. This will be used if the location found in
# a tweet cannot be located in the gazetteer to return coordinate values
geolocator = Nominatim() 
    
############################################################################################
######## FIND LAT & LONG VALUES FOR CITIES IN GAZETTEER & STORE IN DICTIONARIES ############
############################################################################################

# the purpose of the entire for loop is to iterate through each location in the set cities2
# identify the coordinate values and store them in their respective dictionaries

print("find coordinates of cities")
      
# for each city in the set 'cities2', find the lat & long values, populate values in dictionaries 
for a in cities2:
    # if a city location is populated in the set 'cities2' (not 'N/A')
    if a != 'N/A':
        # if the dictionary value is not empty (the location's lat & long are known), pass
        # you pass since there is no need to store the values in the dictionary (they're known)
        # the dictionary will be empty initially but will be updated after each loop
        if city_lat_dic.get(a) != None and city_long_dic.get(a) != None:
            pass
        else:
            # try and obtain the lat & long for a city from the offline gazetteer
            try:
                # find the name of the city in the set in the offline gazetteer & return the latitude
                latitude = gaz.loc[gaz['Name'] == a, 'Latitude']
                # obtain the first element and store as a string
                latitude = str(latitude.iloc[0])
                # find the name of the city in the set in the offline gazetteer & return the longitude
                longitude = gaz.loc[gaz['Name'] == a, 'Longitude']
                # obtain the first element and store as a string
                longitude = str(longitude.iloc[0])
            # if the .loc function is unsuccessful and returns values, pass
            except:
                pass
            # check if the values returned from the gazetteer are not blank for both lat & long
            # if a value is returned, populate the lat & long in their respective dictionaries
            try:   
                if latitude != '' and longitude != '':
                    city_lat_dic[a] = latitude
                    city_long_dic[a] = longitude
                # if no value is returned from the gazetteer, search for lat & long using 
                # an online API geocoding service called 'geolocator'
                else:
                    # return the location coordinates returned from the geolocator function
                    # this is stored as a tree called 'location'
                    location = geolocator.geocode(a)
            # if the online function returns an error or is unsuccesful, pass
            except:
                pass
            try:
                # if both lat & long values returned from the function are not blank, populate 
                # the values retrieved in the respective lat & long dictionaries
                if location.latitude != '' and location.longitude != '':                
                    city_lat_dic[a] = location.latitude
                    city_long_dic[a] = location.longitude
                # if both lat & long are blank, populate both dictionaries with 'N/A'
                else:
                    city_lat_dic[a] = 'N/A'
                    city_long_dic[a] = 'N/A'
            except:
                pass
    # populate the dictionary with 'N/A', since both the gazetteer and online API cannot find its
    # lat & long               
    else:
        city_lat_dic[a] = 'N/A'
        city_long_dic[a] = 'N/A'

print("found city coordinates")
                
############################################################################################
######### JOIN LAT & LONG OF CITY RESULTS TO RESPECTIVE TWEETS IN A DATAFRAME ##############
############################################################################################
  
# create new lists to store the lat & long values for each city        
city_lat = []
city_long = []
                
# the lat & long values of cities will be added to their respective lists, as they currently exist
# in dictionaries. Numerical values will be populated if successful, otherwise 'N/A'

# loop through the list of locations in 'cities_new' (locations found in processed tweets)
# obtain the lat & long returned from the gazetteer & online API, located in the dictionaries 
for a in cities_new:
    city_lat.append(city_lat_dic.get(a))
    city_long.append(city_long_dic.get(a))
    
# turn lists into dataframes
city_lat = pd.DataFrame(city_lat, columns=['city_lat'])
city_long = pd.DataFrame(city_long, columns=['city_long']) 

# join lat & long dataframes with dataframe that contains tweets with locations in body (city_country)
city_country = pd.concat([city_country, city_lat,city_long], axis=1)

# drop any rows that contain 'None' in 'city_lat' & 'city_long' columns
city_country = city_country.replace(to_replace='None', value= np.nan).dropna()

############################################################################################
###### FIND LAT & LONG VALUES FOR COUNRTRIES IN GAZETTEER & STORE IN DICTIONARIES ##########
############################################################################################

# the purpose of the entire for loop is to iterate through each location in the set countries2
# identify the coordinate values and store them in their respective dictionaries

# create new dictionaries to populate the coordinates for countries found in the offline gazetteer
country_lat_dic = {}
country_long_dic = {}

print("find country coordinates")

# for each country in the set 'countries2', find the lat & long, populate values into respective
# dictionaries
for a in countries2:
    # if a country location is populated in the set 'country2' (not 'N/A')
    if a != 'N/A':
        # if the dictionary value is not empty (the location's lat & long are known), pass
        # you pass since there is no need to store the values in the dictionary (they're known)
        # the dictionary will be empty initially but will be updated after each loop
        if country_lat_dic.get(a) != None and country_long_dic.get(a) != None:
            pass
        else:
            # try and obtain the lat & long for a city from the offline gazetteer
            try:
                # find the name of the country in the set, in the offline gazetteer & return the latitude
                latitude = gaz.loc[gaz['Name'] == a, 'Latitude']
                # obtain the first element and store as a string
                latitude = str(latitude.iloc[0])
                # find the name of the country in the set, in the offline gazetteer & return the longitude
                longitude = gaz.loc[gaz['Name'] == a, 'Longitude']
                # obtain the first element and store as a string
                longitude = str(longitude.iloc[0])
            # if the .loc function is unsuccessful and returns values, pass
            except:
                pass
            # check if the values returned from the gazetteer are not blank for both lat & long
            # if a value is returned, populate the lat & long in their respective dictionaries
            try:   
                if latitude != '' and longitude != '':
                    country_lat_dic[a] = latitude
                    country_long_dic[a] = longitude
                # if no value is returned from the gazetteer, search for lat & long using 
                # an online API geocoding service called 'geolocator'
                else:
                    # return the location coordinates returned from the geolocator function
                    location = geolocator.geocode(a)
            # if the online function returns an error or is unsuccesful, pass
            except:
                pass
            try:
                # if both lat & long values returned from the function are not blank, populate 
                # the values retrieved in the respective lat & long dictionaries
                if location.latitude != '' and location.longitude != '':                
                    country_lat_dic[a] = location.latitude
                    country_long_dic[a] = location.longitude
                # if both lat & long are blank, populate both dictionaries with 'N/A'
                else:
                    country_lat_dic[a] = 'N/A'
                    country_long_dic[a] = 'N/A'
            except:
                pass
    # populate the dictionary with 'N/A', since both the gazetteer and online API cannot find its
    # lat & long               
    else:
        country_lat_dic[a] = 'N/A'
        country_long_dic[a] = 'N/A'
        
print("found country coordinates")
       
############################################################################################
############### JOIN LAT & LONG RESULTS OF COUNTRIES TO RESPECTIVE TWEETS ##################
############################################################################################

# create new dictionaries to populate the coordinates for countries found in the offline gazetteer
country_lat = []
country_long = []
        
# loop through the list of locations in 'countries' (locations found in processed tweets)
# obtain the lat & long returned from the gazetteer & online API, located in the dictionaries 
for a in countries:
    country_lat.append(country_lat_dic.get(a))
    country_long.append(country_long_dic.get(a))
    
# turn lists into dataframes
country_lat = pd.DataFrame(country_lat, columns=['country_lat'])
country_long = pd.DataFrame(country_long, columns=['country_long']) 

# concatenate processed tweet, city and country dataframes together
city_country = pd.concat([city_country, country_lat,country_long], axis=1)

# before dropping irrelevant columns from the 'city_country' dataframe, so that only city or
# country coordinates exist for a tweet (not both), store the number of tweets for different
# combinations

### city related values ###

# store number of tweets containing a city, no country but no lat and long found for city
num_tweets_city_nocoord_no_country = len(city_country[(city_country.city != "N/A") &
                                                      (city_country.country == "N/A") &
                                                      (city_country.city_lat == "N/A") & 
                                                      (city_country.city_long == "N/A")])
  
# store number of tweets containing a city and a country but no lat and long found for city
num_tweets_city_nocoord_with_country = len(city_country[(city_country.city != "N/A") &
                                                      (city_country.country != "N/A") &
                                                      (city_country.city_lat == "N/A") & 
                                                      (city_country.city_long == "N/A")])
    
# store number of tweets containing a city, no country and lat and long found for city
num_tweets_city_coord_no_country = len(city_country[(city_country.city != "N/A") &
                                                      (city_country.country == "N/A") &
                                                      (city_country.city_lat != "N/A") & 
                                                      (city_country.city_long != "N/A")])
    
# store number of tweets containing a city and a country and lat or long found for city
num_tweets_city_coord_with_country = len(city_country[(city_country.city != "N/A") &
                                                      (city_country.country != "N/A") &
                                                      (city_country.city_lat != "N/A") & 
                                                      (city_country.city_long != "N/A")])
    
### country related values ###    
    
# store number of tweets containing a country, no city but no lat and long found for country
num_tweets_country_nocoord_no_city = len(city_country[(city_country.country != "N/A") &
                                                      (city_country.city == "N/A") &
                                                      (city_country.country_lat == "N/A") & 
                                                      (city_country.country_long == "N/A")])
  
# store number of tweets containing a country and a city but no lat and long found for country
num_tweets_country_nocoord_with_city = len(city_country[(city_country.country != "N/A") &
                                                      (city_country.city != "N/A") &
                                                      (city_country.country_lat == "N/A") & 
                                                      (city_country.country_long == "N/A")])
    
# store number of tweets containing a country, no city and lat and long found for country
num_tweets_country_coord_no_city = len(city_country[(city_country.country != "N/A") &
                                                      (city_country.city == "N/A") &
                                                      (city_country.country_lat != "N/A") & 
                                                      (city_country.country_long != "N/A")])
    
# store number of tweets containing a country and a city and lat or long found for country
num_tweets_country_coord_with_city = len(city_country[(city_country.country != "N/A") &
                                                      (city_country.city != "N/A") &
                                                      (city_country.country_lat != "N/A") & 
                                                      (city_country.country_long != "N/A")])  

# drop any tweets, where any column contains 'NaN' (not "N/A") from the 'city_country' dataframe    
city_country = city_country.dropna()

# add 2 new columns to the 'city_country' dataframe by appending another df with 2 empty columns
# populate the 2 columns ('lat' & 'long' with city values if they exist, otherwise country)
empty_columns = pd.DataFrame(columns = ['lat', 'long'])

city_country = pd.concat([city_country, empty_columns], axis=1)

# if city coordinates exist, populate them in the 'lat' & 'long' columns, otherwise country coordinates
# city coordinates are more specific than country coordinates, hence they are prioritised
city_country['lat'] = np.where(city_country['city_lat'] == 'N/A', 
            city_country['country_lat'], city_country['city_lat'])

city_country['long'] = np.where(city_country['city_long'] == 'N/A', 
            city_country['country_long'], city_country['city_long'])

# reset the index of 'city_country' after dropping all rows that contain 'NaN
city_country = city_country.reset_index()
city_country = city_country.drop(['index'], axis=1)

# drop the lat, long, city & country values for both cities and countries, now a new columns
# have been created that contain city coordinates if they exist, otherwise country coordinates
city_country = city_country.drop(['country', 'city', 'city_lat', 'city_long',
                                  'country_lat', 'country_long'], axis=1)
    
# append tweets that were geotagged (agg_cord) with tweets that were not geotagged but have
# since had their lat & long values identified through geotagging, then reset index
city_country = city_country.append(agg_cord)
city_country = city_country.reset_index()
city_country = city_country.drop(['index'], axis=1)

############################################################################################
############################ EXPORT STATISTICS FOR ANALYSIS $##############################
############################################################################################

print("export stats")

# the number of tweets at different stages of geotagging and their results are joined to form
# a dataframe and exported as a .csv

stats_export = pd.DataFrame([{'num_tweets_imported':num_tweets_imported,
                              'num_geotagged_tweets':num_geotagged_tweets,
                              'num_tweets_nocity_nocountry':num_tweets_nocity_nocountry,    
                              'num_non_geotagged_tweets':num_non_geotagged_tweets,
                              'num_tweets_city_coord_no_country':num_tweets_city_coord_no_country,
                              'num_tweets_city_coord_with_country':num_tweets_city_coord_with_country,
                              'num_tweets_city_no_country':num_tweets_city_no_country,
                              'num_tweets_city_nocoord_no_country':num_tweets_city_nocoord_no_country,
                              'num_tweets_city_nocoord_with_country':num_tweets_city_nocoord_with_country,
                              'num_tweets_city_or_country_or_both':num_tweets_city_or_country_or_both,
                              'num_tweets_country_and_city':num_tweets_country_and_city,
                              'num_tweets_country_coord_no_city':num_tweets_country_coord_no_city,
                              'num_tweets_country_coord_with_city':num_tweets_country_coord_with_city,
                              'num_tweets_country_no_city':num_tweets_country_no_city,
                              'num_tweets_country_nocoord_no_city':num_tweets_country_nocoord_no_city,
                              'num_tweets_country_nocoord_with_city':num_tweets_country_nocoord_with_city}])
                             
stats_export.to_csv("stats_export.csv", index=False)

print("stats exported")
    
############################################################################################
####################### CONVERT EVENT TYPES INTO NUMERICAL VALUES ##########################
############################################################################################

# represent event type (e.g. earthquake or typhoon) as an integer 
city_country['event_id'] = city_country['event_type'].factorize()[0]

# create new dataframe 'event_id_df' from 'city_country' and remove any duplicates
# this is used for mapping an 'event_type' to 'event_id'
event_id_df = city_country[['event_type', 'event_id']].drop_duplicates()

# reset the index of 'event_id_df' dataframe and drop the new column created
event_id_df = event_id_df.reset_index()
event_id_df = event_id_df.drop(['index'], axis=1)

### create dictionaries for future 'vlookup' function ###

# key = event name, value = event_id
event_to_id = dict(event_id_df.values)

# key = event id, value = event name
id_to_event = dict(event_id_df[['event_id','event_type']].values)
    
############################################################################################
##################### CLASSIFY TWEETS BY EVENT TYPE (E.G. TYPHOON) #########################
############################################################################################

#import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict

# create an array object containing the 'event_id' from the dataframe 'city_country'
# this is the y 'label' used for classification
y_label = city_country.iloc[:,:].values
y_label = y_label[:,4]
  
########## Use tf-idf to vectorise the frequency of words in processed tweets ###################

# sublinear_df = true => logarithmic form of frequency
# min_df = minimum number of tweets a word must be present in to be kept => 0
# norm set to l2 to ensure all feature vectors have a Euclidian norm of 1
# ngram_range - changed to analyse unigrams, bigrams and trigrams
# stop_words remove all common pronouns, which should already have been done => reduce noisy features (high frequency)

print("vectorise all tweets")

# vectorize for unigrams (words)
tfidf_words = TfidfVectorizer(sublinear_tf=True, min_df=0, norm='l2', encoding='latin-1',
                              ngram_range=(1,1), stop_words='english', max_features=10000)

## vectorize for bigrams
#tfidf_bigrams = TfidfVectorizer(sublinear_tf=True, min_df=0, norm='l2', encoding='latin-1',
#                              ngram_range=(2,2), stop_words='english', max_features=10000)
#
## vecorize for trigrams
#tfidf_trigrams = TfidfVectorizer(sublinear_tf=True, min_df=0, norm='l2', encoding='latin-1',
#                              ngram_range=(3,3), stop_words='english', max_features=10000)

print("fit vectorised tweets")

x_words = tfidf_words.fit_transform(city_country.preprocessed).toarray()
#x_bigrams = tfidf_bigrams.fit_transform(city_country.preprocessed).toarray()
#x_trigrams = tfidf_trigrams.fit_transform(city_country.preprocessed).toarray()
   
############################################################################################
################################## FIT AND TEST CLASSIFIERS ################################
############################################################################################

print("define models")

# define models to be evaluated
svm = OneVsRestClassifier(LinearSVC(C=0.9, random_state=0))
nb = OneVsRestClassifier(MultinomialNB(alpha=0.9))
lr = OneVsRestClassifier(LogisticRegression(C=0.9, random_state=0))
rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100,criterion="entropy",
                                                max_depth=6, random_state=0))

print("models defined")

########## Fitting, predicting and calculating average accuracy for unigrams data ##########

# libraries used for ROC curves and accuracy
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import accuracy_score

print("begin predictions")

y_label = pd.Series(y_label.tolist())

# 10-fold cross validation using stratified k sampling for each of the models being tested
# the stratified k sampling is the default when you set cv to an integer in the function
cross_val_predict_svm = cross_val_predict(svm,x_words, y_label, cv=10)
cross_val_predict_nb = cross_val_predict(nb,x_words, y_label, cv=10)
cross_val_predict_lr = cross_val_predict(lr,x_words, y_label, cv=10)
cross_val_predict_rf = cross_val_predict(rf,x_words, y_label, cv=10)

print("finished predictions")

# convert the predicted event_ids into dataframes
cross_val_predict_svm = pd.DataFrame(cross_val_predict_svm)
cross_val_predict_svm.columns = ['predicted_event_id']

cross_val_predict_nb = pd.DataFrame(cross_val_predict_nb)
cross_val_predict_nb.columns = ['predicted_event_id']

cross_val_predict_lr = pd.DataFrame(cross_val_predict_lr)
cross_val_predict_lr.columns = ['predicted_event_id']

cross_val_predict_rf = pd.DataFrame(cross_val_predict_rf)
cross_val_predict_rf.columns = ['predicted_event_id']

# join the 'cross_val_predict' dataframe with the 'x_train_tweets' dataframe, which contains 
# tweet, actual event-type, actual event_id, lat & long values 
cross_val_predict_svm = pd.concat([city_country, cross_val_predict_svm], axis=1)
cross_val_predict_nb = pd.concat([city_country, cross_val_predict_nb], axis=1)
cross_val_predict_lr = pd.concat([city_country, cross_val_predict_lr], axis=1)
cross_val_predict_rf = pd.concat([city_country, cross_val_predict_rf], axis=1)

# list of all of the dataframes that contain predictions (will be iterated over)
cross_val_list = [cross_val_predict_svm,cross_val_predict_nb,cross_val_predict_lr,cross_val_predict_rf]

# for each dataframe that contains predictions (1 per model evaluated)
for cross_pred in cross_val_list:
    # map the event_type in the dataframe 'cross_val_svm' for predicted and actual event type
    for index, row in cross_pred.iterrows():
        cross_pred.at[index,'predicted_event_type'] = id_to_event.get(cross_pred.predicted_event_id[index])
    
    # re-order columns in cross_pred dataframe
    cross_pred = cross_pred[['preprocessed','lat','long','predicted_event_id',
                             'event_id','predicted_event_type','event_type']]

# Compute ROC curve and AUC for each class
                
# create list that contains the dataframes of the predictions from each model tested
cross_val_list = [cross_val_predict_svm,cross_val_predict_nb,cross_val_predict_lr,cross_val_predict_rf]

# create list of the names of models tested (used for creating labels on the ROC curves)
cross_val_names = ['SVC','Naive Bayes','Logistic Regression','Random Forest']

print("calculate accuracies")

### calculate accuracies of each model ###        
accuracy_svm = accuracy_score(cross_val_predict_svm.event_id, cross_val_predict_svm.predicted_event_id)
accuracy_nb = accuracy_score(cross_val_predict_nb.event_id, cross_val_predict_nb.predicted_event_id)
accuracy_lr = accuracy_score(cross_val_predict_lr.event_id, cross_val_predict_lr.predicted_event_id)
accuracy_rf = accuracy_score(cross_val_predict_rf.event_id, cross_val_predict_rf.predicted_event_id)

print("accuracies calculated")

# convert variables into a dataframe (SVC, NB, Logistic Regression & Random Forest)
accuracy_svm = [accuracy_svm]
accuracy_svm = pd.DataFrame(accuracy_svm)
accuracy_svm.columns = ['accuracy_svm']

accuracy_nb = [accuracy_nb]
accuracy_nb = pd.DataFrame(accuracy_nb)
accuracy_nb.columns = ['accuracy_nb']

accuracy_lr = [accuracy_lr]
accuracy_lr = pd.DataFrame(accuracy_lr)
accuracy_lr.columns = ['accuracy_lr']

accuracy_rf = [accuracy_rf]
accuracy_rf = pd.DataFrame(accuracy_rf)
accuracy_rf.columns = ['accuracy_rf']

# concatenate the accuracies of the accuracies into 1 dataframe 
accuracies_all_models = pd.concat([accuracy_svm, accuracy_nb, accuracy_lr, accuracy_rf], axis=1 )

filename = "accuracies_all_models_unigrams.csv"
# export dataframe as a .csv file
accuracies_all_models.to_csv(filename, index=False)
   
# initialise b. this is used to iterate through the names of each model 
b = 0

print("begin producing ROC curves")

# create ROC curves and the AUC for each model tested
# for each dataframe that contains the actual and predicte event ids
for cross_pred in cross_val_list:
    class_list = range(len(event_to_id))
    
    y_score = label_binarize(cross_pred.predicted_event_id, classes=class_list)
    y_test_1 = label_binarize(cross_pred.event_id, classes=class_list)
    
    n_classes = y_score.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_1[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in class_list]))
    
    # Interpolate all ROC curves at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in class_list:
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["macro"]), color='green', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, colour in zip(class_list, colors):
        plt.plot(fpr[i], tpr[i], color=colour, label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
        
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For ' + str(cross_val_names[b]) + '_unigrams')
    plt.legend(loc="lower right")
    plt.savefig('ROC_' + str(cross_val_names[b]) + '_unigrams.png')
    plt.show()
    plt.close()
    
    b = b+1
    
print("ROC curves finished")


     
