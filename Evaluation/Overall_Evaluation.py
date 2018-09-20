# import libraries
from geotext import GeoText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import Nominatim
import random

############################################################################################
############ IMPORT RANDOMLY SELETED PERCENTAGE OF REDUCED CORPUS ##############
################################################################################

# import a randomly selected percentage of the corpus
# adjust the p value to the percentage required, e.g. p = 0.2 is 20% of corpus

filename = "agg_reduced.csv"
p = 0.2   # % of the lines
# keep the header, then take only % of lines
# if random from [0,1] interval is > than %, the row will be skipped
agg = pd.read_csv(filename, header=0, skiprows=lambda i: i>0 and random.random() > p)

print("data imported")

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

print("start identifying locations")

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
        
print("finished identifying locations")

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

print("open gazetteer")


# load an offline gazetteer for offline geocoding 
gaz = pd.read_csv('gaz.csv')

print("gazetteer open")

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

print("find city coordinates")

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
                
############################################################################################
######### JOIN LAT & LONG OF CITY RESULTS TO RESPECTIVE TWEETS IN A DATAFRAME ##############
############################################################################################
  
print("found city coordinates")    
    
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
       
############################################################################################
############### JOIN LAT & LONG RESULTS OF COUNTRIES TO RESPECTIVE TWEETS ##################
############################################################################################

print("found country coordinates")   

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

print("print stats file")   

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
############################## VECTORISE TWEETS USING TF-IDF ###############################
############################################################################################

#import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier

# create an array object containing the 'event_id' from the dataframe 'city_country'
# this is the y 'label' used for classification
y_label = city_country.iloc[:,:].values
y_label = y_label[:,4]
  
########## Use tf-idf to vectorise the frequency of words in processed tweets ###############

# sublinear_df = true => logarithmic form of frequency
# min_df = minimum number of tweets a word must be present in to be kept => 0
# norm set to l2 to ensure all feature vectors have a Euclidian norm of 1
# ngram_range - changed to analyse unigrams, bigrams and trigrams
# stop_words remove all common pronouns => reduce noisy features (high frequency)
# max_features = 10000 (computational cost is too expensive if any larger)
# max_df = remove words appearing in more than 80% of tweets (advised in the package documentation)

print("start vectorise")   

# vectorize for unigrams (words)
tfidf_words = TfidfVectorizer(sublinear_tf=True, min_df=0, norm='l2', encoding='latin-1',
                              ngram_range=(1,1), stop_words='english', max_features=10000,
                              max_df=1)

print("finish vectorise")   

# transform the fitted tf-idf vectors and transform to an array
# all three use 'preprocessed', which are the tweets from the 'city_country' dataframe

print("fit vector")   

x_words = tfidf_words.fit_transform(city_country.preprocessed).toarray()

print("vector fitted")   

############################################################################################
############################## CREATE TRAINING AND TEST DATA ###############################
############################################################################################

# split the full dataset into training and test (80%/20%, respectively)

# unigrams only

print("split unigrams into train and test")   

x_train_words, x_test_words, y_train, y_test = train_test_split(x_words, y_label, 
                                                            test_size = 0.20, random_state = 0)

# mimic the split with the raw data, so that the tweets can be appended to the predictions later
# these are not used for prediction purposes, so no unnecessary columns are removed
print("split city_country into train and test")   

x_train_tweets, x_test_tweets, y_train_tweets, y_test_tweets = train_test_split(city_country, y_label, 
                                                            test_size = 0.20, random_state = 0)

# turn 'x_train_tweets' & 'x_test_tweets' into dataframes & reset their indices
x_train_tweets = pd.DataFrame(x_train_tweets)
x_train_tweets = x_train_tweets.reset_index()
x_train_tweets = x_train_tweets.drop(['index'], axis=1)

x_test_tweets = pd.DataFrame(x_test_tweets)
x_test_tweets = x_test_tweets.reset_index()
x_test_tweets = x_test_tweets.drop(['index'], axis=1)

############################################################################################
###################################  OVERSAMPLE WITH SMOTE #################################
############################################################################################

print("start smote")

# Use SMOTE to oversample the minority classes
from imblearn.over_sampling import SMOTE

# define function
sm = SMOTE(random_state=12)

# perform SMOTE on training sets (vectorised tweets and normal tweets)
# test sets are not ovesample as this would artificially boost the number of tweets, which is a key
# input in the DBSCAN algorithm performed later to detect a disaster
x_train_words_sm, y_train_words_sm = sm.fit_sample(x_train_words, y_train)

print("finish smote")

# count the number of classes that exist before and after SMOTE, to ensure its implemented correctly
from collections import Counter
class_check_train_woSMOTE = Counter(y_train)
class_check_train_SMOTE = Counter(y_train_words_sm)

# 'y_train_words_sm' converted to series (trainingset of vectorised tweets for prediction)
y_train_words_sm = pd.Series(y_train_words_sm.tolist())

############################################################################################
######## EVALUATE THE PERFORMANCE OF BEST PERFORMING CLASSIFIER IN MORE DETAIL #############
############################################################################################

#################### FIT AND EVALUATE LINEAR SVC (BEST PERFORMING MODEL) ###################

# select and run a linear Support Vector Classifier (the best performing classifier)
svc_model = OneVsRestClassifier(LinearSVC(C=0.375))

# fit the SVC model with the oversampled vectorised dataset using tf-idf
print("fit svc ")   

svc_model.fit(x_train_words_sm, y_train_words_sm)

# predict using the unigrams test data transformed using tf-idf
# the test data has not been oversampled, as this would artificially boost the number of tweets
# which is a key input in the DBSCAN algorithm performed later to detect a disaster
print("predict using svc ")   

predictions_svc = svc_model.predict(x_test_words)

# convert the predicted event_ids into dataframes
predictions_svc = pd.DataFrame(predictions_svc)
predictions_svc.columns = ['predicted_event_id']

# join the 'predictions_svc' df with the 'x_test' df, which contains the tweet's actual event-type, 
# actual event_id, lat & long values (will be used for comparisons of predictions)
predictions_svc = pd.concat([x_test_tweets, predictions_svc], axis=1)

print("map event_id for pred and actual event_type")

# map the event_type in the dataframe 'predictions_svc' for predicted and actual event_type
for index, row in predictions_svc.iterrows():
    predictions_svc.at[index,'predicted_event_type'] = id_to_event.get(predictions_svc.predicted_event_id[index])

# re-order columns in predictions_svc dataframe
predictions_svc = predictions_svc[['preprocessed','lat','long','predicted_event_id',
                         'event_id','predicted_event_type','event_type']]

########################### FIT AND EVALUATE NAIVE BAYES (BENCHMARK MODEL) #######################

# Select a Naive Bayes model for benchmarking purposes
nb_model = OneVsRestClassifier(MultinomialNB(alpha=0.1))

# fit the NB model with the oversampled vectorised dataset using tf-idf
print("fit nb ")   

nb_model.fit(x_train_words_sm, y_train_words_sm)

# predict using the unigrams test data transformed using tf-idf
# the test data has not been oversampled, as this would artificially boost the number of tweets
# which is a key input in the DBSCAN algorithm performed later to detect a disaster
print("predict using nb ")   

predictions_nb = nb_model.predict(x_test_words)

# convert the predicted event_ids into dataframes
predictions_nb = pd.DataFrame(predictions_nb)
predictions_nb.columns = ['predicted_event_id']

# join the 'predictions_nb' df with the 'x_test' df, which contains the tweet's actual event-type, 
# actual event_id, lat & long values (will be used for comparisons of predictions)
predictions_nb = pd.concat([x_test_tweets, predictions_nb], axis=1)

print("map event_id for pred and actual event_type")

# map the event_type in the dataframe 'predictions_nb' for predicted and actual event_type
for index, row in predictions_nb.iterrows():
    predictions_nb.at[index,'predicted_event_type'] = id_to_event.get(predictions_nb.predicted_event_id[index])

# re-order columns in predictions_nb dataframe
predictions_nb = predictions_nb[['preprocessed','lat','long','predicted_event_id',
                         'event_id','predicted_event_type','event_type']]

########################### CALCULATE ACCURACIES FOR SVC & NB MODELS & EXPORT #######################

from sklearn.metrics import accuracy_score

# calculate the accuracy for SVM and NB
print("calculate accuracies of svm and nb")   

accuracy_svc = accuracy_score(predictions_svc.event_id, predictions_svc.predicted_event_id)
accuracy_nb = accuracy_score(predictions_nb.event_id, predictions_nb.predicted_event_id)

# convert variable into a dataframe
accuracy_svc = [accuracy_svc]
accuracy_svc = pd.DataFrame(accuracy_svc)
accuracy_svc.columns = ['accuracy_svc']

# convert variable into a dataframe
accuracy_nb = [accuracy_nb]
accuracy_nb = pd.DataFrame(accuracy_nb)
accuracy_nb.columns = ['accuracy_nb']

# concatenate the accuracies of the SVM and NB models 
accuracies_svm_and_nb = pd.concat([accuracy_svc, accuracy_nb], axis=1 )

accuracies_svm_and_nb.to_csv("accuracies_svm_and_nb.csv", index=False)

print("accuracies calculated")

############################################################################################
############################# CONSTRUCT A CONFUSION MATRIX #################################
############################################################################################

from sklearn.metrics import confusion_matrix

# Construct a confusion matrix for svm
print("calculate confusion matrices")   

conf_mat_svc = confusion_matrix(predictions_svc.event_id, predictions_svc.predicted_event_id)
plt.figure()
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(conf_mat_svc, annot=True, fmt='d', xticklabels=event_id_df.event_type.values, 
            yticklabels=event_id_df.event_type.values)
plt.ylabel('Actual')
plt.xlabel('Predicted_SVC')
plt.savefig('confusion_matrix_svc.png')
plt.close()

# Construct a confusion matrix for nb
conf_mat_nb = confusion_matrix(predictions_nb.event_id, predictions_nb.predicted_event_id)
plt.figure()
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(conf_mat_nb, annot=True, fmt='d', xticklabels=event_id_df.event_type.values, 
            yticklabels=event_id_df.event_type.values)
plt.ylabel('Actual')
plt.xlabel('Predicted_NB')
plt.savefig('confusion_matrix_nb.png')
plt.close()

############################################################################################
############################### REVIEW MISCLASSIFIED DATA ##################################
############################################################################################

# Take a look the misclassified data for SVC. 
# Threshold has been manually set to 3, which can be adjusted

print("show misclassified tweets")   
            
misclassified_svc = []
misclassified_svc_details = []

for predicted in event_id_df.event_id:
    for actual in event_id_df.event_id:
        if predicted != actual and conf_mat_svc[actual, predicted] >= 3:
            misclassified_svc.append("'{}' predicted as '{}' : {} examples.".format(id_to_event[actual], id_to_event[predicted], 
                  conf_mat_nb[actual,predicted]))
            misclassified_svc_details.append(predictions_svc.loc[predictions_svc.index[(predictions_svc.event_id == actual)& (predictions_svc.predicted_event_id == predicted)]][['event_type', 'preprocessed']])

# Create new dataframe to store all misclassified tweets 
all_misclassified_tweets_svc = pd.DataFrame()

# iterate through each dataframe in the list 'misclassified_svm_details' and concatenate
for dataframe in misclassified_svc_details:
	# Iteratively add each dataframe to previous results
     all_misclassified_tweets_svc = pd.concat([all_misclassified_tweets_svc, dataframe])

all_misclassified_tweets_svc.to_csv("all_misclassified_tweets_svc.csv", index=False)   
    
misclassified_nb = []
misclassified_nb_details = []

for predicted in event_id_df.event_id:
    for actual in event_id_df.event_id:
        if predicted != actual and conf_mat_nb[actual, predicted] >= 3:
            misclassified_nb.append("'{}' predicted as '{}' : {} examples.".format(id_to_event[actual], id_to_event[predicted], 
                  conf_mat_nb[actual,predicted]))
            misclassified_nb_details.append(predictions_nb.loc[predictions_nb.index[(predictions_nb.event_id == actual)& (predictions_nb.predicted_event_id == predicted)]][['event_type', 'preprocessed']])

# Create new dataframe to store all misclassified tweets 
all_misclassified_tweets_nb = pd.DataFrame()

# iterate through each dataframe in the list 'misclassified_svm_details' and concatenate
for dataframe in misclassified_nb_details:
	# Iteratively add each dataframe to previous results
     all_misclassified_tweets_nb = pd.concat([all_misclassified_tweets_nb, dataframe])

all_misclassified_tweets_nb.to_csv("all_misclassified_tweets_nb.csv", index=False)   
                        
############################################################################################
########################### COMPUTE ROC CURVE AND AUC FOR SVC $#############################
############################################################################################
            
# Compute ROC curve and AUC for each class

print("calculate ROC curves")   
            
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

class_list = range(len(event_to_id))

y_score = label_binarize(predictions_svc.predicted_event_id, classes=class_list)
y_test_1 = label_binarize(predictions_svc.event_id, classes=class_list)

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
plt.title('Receiver Operating Characteristic (ROC) For SVC')
plt.legend(loc="lower right")
plt.savefig('ROC_svc.png')
plt.show()
plt.close()

############################################################################################
########################### COMPUTE ROC CURVE AND AUC FOR NB $##############################
############################################################################################

# Compute ROC curve and AUC for each class
            
class_list = range(len(event_to_id))

y_score = label_binarize(predictions_nb.predicted_event_id, classes=class_list)
y_test_1 = label_binarize(predictions_nb.event_id, classes=class_list)

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
plt.title('Receiver Operating Characteristic (ROC) For Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('ROC_nb.png')
plt.show()
plt.close()

#############################################################################################
############################# DBSCAN CLUSTERING & EXPORTING #################################
#############################################################################################

print("start DBSCAN section")   

event_types = ['earthquake','typhoon','shooting','cyclone','plane_crash','hurricane',
               'volcano_eruption','flood','school_attack']

# DEFINE THE CLUSTERING FUNCTIONS ##

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
  
# find the point in each cluster that is closest to its centroid
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

# define the number of KMs in one radian
kms_per_radian = 6371.0088

epsilon = 20 / kms_per_radian

names_of_centermost_points = ['points_earth', 'points_typh', 'points_shoot', 'points_cyclone',
                              'points_plane_crash','points_hurricane', 'points_volcano_eruption',
                              'points_flood','points_school_attack']

################################# DBSCAN for SVC. Each event_type #############################

print("SVC for loop for DBSCAN")   

a=0 
# for each event_type for SVC, create a dataframe of the predicted values & reset index
# then perform DBSCAN with the dataframe
for event in event_types:
    event_svc = predictions_svc[predictions_svc['predicted_event_type'] == event_types[a]]
    event_svc = event_svc.reset_index()
    event_svc = event_svc.drop(['index'], axis=1)
    # now perform DBSCAM on the new dataframe 'event_svm' that is overwritten each time
    ### clustering for earthquakes ###  
    coords = event_svc.as_matrix(columns=['lat','long'])
    coords = coords.astype(float)
    # make db = None. If there is sufficient points for DBSCAN to work, it will be overwritten
    # with the results from DBSCAN, otherwise it will remain as None
    # this is subsequently used to errors and the code to run if there are no results from DBSCAN
    db = None
    # try to run the DBSCAN function. if insufficient data, pass (the dataframe remains = None)
    try:
        db = DBSCAN(eps=epsilon, min_samples=20, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    except:
        pass
    # if DBSCAN ran and the dataframe did not remain as 'None', obtain the clusters
    if db != None:
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels)) -1
        print('Number of clusters in ' + str(names_of_centermost_points[a]) + '_svc' + ': {:,}'.format(num_clusters))
        # find the point in each cluster that is closest to its centroid
        clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
        # find the point in each cluster that is closest to its centroid
        centermost_points = clusters.map(get_centermost_point)  
    # if DBSCAN did not run and the dataframe remained as 'None', set the dataframe to 'None'
    # this means that no file will be exported in the following stage
    else:
        centermost_points = None
    ## now extract the centermost points and write to a .csv file
    b = 0
    try:    
        if centermost_points != None:
            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
            rep_points = rep_points.astype(float)
            filename = names_of_centermost_points[a] + str(len(agg)) + "_svc.csv"
            rep_points.to_csv(filename, index=False)
        else:
            pass
    except:
        if len(centermost_points) != 0:
            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
            rep_points = rep_points.astype(float)
            filename = names_of_centermost_points[a] + str(len(agg)) + "_svc.csv"
            rep_points.to_csv(filename, index=False)
        else:
            pass
    a = a+1


###################################### PLOT CLUSTERS FOR SVC ###################################

from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import pandas as pd, numpy as np, matplotlib.pyplot as plt

points = str(len(agg))

#magic command to display matplotlib plots inline within the ipython notebook
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# define map colors
land_color = '#f5f5f3'
water_color = '#cdd2d4'
coastline_color = '#f5f5f3'
border_color = '#bbbbbb'
meridian_color = '#eaeaea'
marker_edge_color = 'None'

#define event colours
typhoon = '#0033cc' # dark blue
earthquake = '#00c5cc' # aqua blue
cyclone = '#cc5100' # orange
plane_crash = '#efef13' # yellow
hurricane = '#55ef13' # light green
volcano_eruption = '#0a5b1f' # dark green
flood = '#ef09d4' # pink
school_attack = '#0e000f' # black

earth_svc = "points_earth" + points + "_svc.csv"
flood_svc = "points_flood" + points + "_svc.csv"
plane_crash_svc = "points_plane_crash" + points + "_svc.csv"
school_attack_svc = "points_school_attack" + points + "_svc.csv"
typh_svc = "points_typh" + points + "_svc.csv"
volcano_svc = "points_volcano_eruption" + points + "_svc.csv"
cyclone_svc = "points_cyclone" + points + "_svc.csv"
hurricane_svc = "points_hurricane" + points + "_svc.csv"


try:
    rep_points_earth_svc = pd.read_csv(earth_svc)
except:
    pass

try:
    rep_points_flood_svc = pd.read_csv(flood_svc)
except:
    pass

try:
    rep_points_plane_crash_svc = pd.read_csv(plane_crash_svc)
except:
    pass

try:
    rep_points_school_attack_svc = pd.read_csv(school_attack_svc)
except:
    pass

try:
    rep_points_typh_svc = pd.read_csv(typh_svc)
except:
    pass

try:
    rep_points_volcano_eruption_svc = pd.read_csv(volcano_svc)
except:
    pass

try:
    rep_points_cyclone_svc = pd.read_csv(cyclone_svc)
except:
    pass

try:
    rep_points_hurricane_svc = pd.read_csv(hurricane_svc)
except:
    pass

# create the plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, facecolor='#ffffff', frame_on=False)
ax.set_title('Central Location of Identified Disaters (SVC)', fontsize=24, color='#333333')

# draw the basemap and its features

m = Basemap(projection='kav7', lon_0=0, resolution='l', area_thresh=10000)
m.drawmapboundary(color=border_color, fill_color=water_color)
m.drawcoastlines(color=coastline_color)
m.drawcountries(color=border_color)
m.fillcontinents(color=land_color, lake_color=water_color)
m.drawparallels(np.arange(-90., 120., 30.), color=meridian_color)
m.drawmeridians(np.arange(0., 420., 60.), color=meridian_color)

#setting the legend for the plot using patch
ty = mpatches.Patch(color = typhoon, label= "Typhoon")
ea = mpatches.Patch(color = earthquake, label = "Earthquake")
cy = mpatches.Patch(color = cyclone, label= "Cyclone")
pl = mpatches.Patch(color = plane_crash, label= "Plane Crash")
hu = mpatches.Patch(color = hurricane, label= "Hurricane")
vo = mpatches.Patch(color = volcano_eruption, label= "Volcano Eruption")
fl = mpatches.Patch(color = flood, label= "Flood")
sc = mpatches.Patch(color = school_attack, label= "School Attack")

plt.legend(handles=[ty,ea,cy,pl,hu,vo,fl,sc],title='Colour of disasters', loc=3)

# project the points from each dataset then concatenate and scatter plot them

# earthquake
try:
    for index, row in rep_points_earth_svc.iterrows():
        rep_points_earth_svc['event_type'] = 'earthquake'
        x_earth_svc,y_earth_svc = m(rep_points_earth_svc['lon'].values, rep_points_earth_svc['lat'].values)
        m.scatter(x_earth_svc, y_earth_svc, s=10, color=earthquake, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass

# typhoon
try:
    for index, row in rep_points_typh_svc.iterrows():
        rep_points_typh_svc['event_type'] = 'typhoon'
        x_typh_svc,y_typh_svc = m(rep_points_typh_svc['lon'].values, rep_points_typh_svc['lat'].values)
        m.scatter(x_typh_svc, x_typh_svc, s=10, color=typhoon, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass

# cyclone 
try:
    for index, row in rep_points_cyclone_svc.iterrows():
        rep_points_cyclone_svc['event_type'] = 'cyclone'
        x_cyclone_svc,y_cyclone_svc = m(rep_points_cyclone_svc['lon'].values, rep_points_cyclone_svc['lat'].values)
        m.scatter(x_cyclone_svc, y_cyclone_svc, s=10, color=cyclone, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# plane_crash    
try:
    for index, row in rep_points_plane_crash_svc.iterrows():
        rep_points_plane_crash_svc['event_type'] = 'plane_crash'
        x_plane_crash_svc,y_plane_crash_svc = m(rep_points_plane_crash_svc['lon'].values, rep_points_plane_crash_svc['lat'].values)
        m.scatter(x_plane_crash_svc, y_plane_crash_svc, s=10, color=plane_crash, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass


# hurricane 
try:
    for index, row in rep_points_hurricane_svc.iterrows():
        rep_points_hurricane_svc['event_type'] = 'hurricane'
        x_hurricane_svc,y_hurricane_svc = m(rep_points_hurricane_svc['lon'].values, rep_points_hurricane_svc['lat'].values)
        m.scatter(x_hurricane_svc, y_hurricane_svc, s=10, color=hurricane, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# volcano    
try:
    for index, row in rep_points_volcano_eruption_svc.iterrows():
        rep_points_volcano_eruption_svc['event_type'] = 'volcano_eruption'
        x_volcano_eruption_svc,y_volcano_eruption_svc = m(rep_points_volcano_eruption_svc['lon'].values, rep_points_volcano_eruption_svc['lat'].values)
        m.scatter(x_volcano_eruption_svc, y_volcano_eruption_svc, s=10, color=volcano_eruption, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# flood    
try:
    for index, row in rep_points_flood_svc.iterrows():
        rep_points_flood_svc['event_type'] = 'flood'
        x_flood_svc,y_flood_svc = m(rep_points_flood_svc['lon'].values, rep_points_flood_svc['lat'].values)
        m.scatter(x_flood_svc, y_flood_svc, s=10, color=flood, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# school_attack    
try:
    for index, row in rep_points_school_attack_svc.iterrows():
        rep_points_school_attack_svc['event_type'] = 'school_attack'
        x_school_attack_svc,y_school_attack_svc = m(rep_points_school_attack_svc['lon'].values, rep_points_school_attack_svc['lat'].values)
        m.scatter(x_school_attack_svc, y_school_attack_svc, s=10, color=school_attack, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# show the map and save the map
plotname = "svc_cluster_plot_" + points + ".png"
plt.savefig(plotname, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

a=0 
# for each event_type for NB, create a dataframe of the predicted values & reset index
# then perform DBSCAN with the dataframe
for event in event_types:
    event_nb = predictions_nb[predictions_nb['predicted_event_type'] == event_types[a]]
    event_nb = event_nb.reset_index()
    event_nb = event_nb.drop(['index'], axis=1)
    # now perform DBSCAM on the new dataframe 'event_svm' that is overwritten each time
    ### clustering for earthquakes ###  
    coords = event_nb.as_matrix(columns=['lat','long'])
    coords = coords.astype(float)
    # make db = None. If there is sufficient points for DBSCAN to work, it will be overwritten
    # with the results from DBSCAN, otherwise it will remain as None
    # this is subsequently used to errors and the code to run if there are no results from DBSCAN
    db = None
    # try to run the DBSCAN function. if insufficient data, pass (the dataframe remains = None)
    try:
        db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    except:
        pass
    # if DBSCAN ran and the dataframe did not remain as 'None', obtain the clusters
    if db != None:
        cluster_labels = db.labels_
        num_clusters = len(set(cluster_labels)) -1
        print('Number of clusters in ' + str(names_of_centermost_points[a]) + '_nb' + ': {:,}'.format(num_clusters))
        # find the point in each cluster that is closest to its centroid
        clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
        # find the point in each cluster that is closest to its centroid
        centermost_points = clusters.map(get_centermost_point)  
    # if DBSCAN did not run and the dataframe remained as 'None', set the dataframe to 'None'
    # this means that no file will be exported in the following stage
    else:
        centermost_points = None
    ## now extract the centermost points and write to a .csv file
    b = 0
    try:    
        if centermost_points != None:
            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
            rep_points = rep_points.astype(float)
            filename = names_of_centermost_points[a] + str(len(agg)) + "_nb.csv"
            rep_points.to_csv(filename, index=False)
        else:
            pass
    except:
        if len(centermost_points) != 0:
            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'lon':lons, 'lat':lats})
            rep_points = rep_points.astype(float)
            filename = names_of_centermost_points[a] + str(len(agg)) + "_nb.csv"
            rep_points.to_csv(filename, index=False)
        else:
            pass
    a = a+1

print("finished")   
    
#############################################################################################
########################### CLUSTERING AND PLOTTING FOR NB ##################################
#############################################################################################

cyclone_nb = "points_cyclone" + points + "_nb.csv"
earth_nb = "points_earth" + points + "_nb.csv"
flood_nb = "points_flood" + points + "_nb.csv"
hurricane_nb = "points_hurricane" + points + "_nb.csv"
plane_crash_nb = "points_plane_crash" + points + "_nb.csv"
school_attack_nb = "points_school_attack" + points + "_nb.csv"
typh_nb= "points_typh" + points + "_nb.csv"
volcano_nb = "points_volcano_eruption" + points + "_nb.csv"

try:
    rep_points_cyclone_nb = pd.read_csv(cyclone_nb)
except:
    pass

try:
    rep_points_earth_nb = pd.read_csv(earth_nb)
except:
    pass

try:
    rep_points_flood_nb = pd.read_csv(flood_nb)
except:
    pass

try:
    rep_points_hurricane_nb = pd.read_csv(hurricane_nb)
except:
    pass

try:
    rep_points_plane_crash_nb = pd.read_csv(plane_crash_nb)
except:
    pass

try:
    rep_points_school_attack_nb = pd.read_csv(school_attack_nb)
except:
    pass

try:
    rep_points_typh_nb = pd.read_csv(typh_nb)
except:
    pass

try:
    rep_points_volcano_eruption_nb = pd.read_csv(volcano_nb)
except:
    pass

#magic command to display matplotlib plots inline within the ipython notebook
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# define map colors
land_color = '#f5f5f3'
water_color = '#cdd2d4'
coastline_color = '#f5f5f3'
border_color = '#bbbbbb'
meridian_color = '#eaeaea'
marker_edge_color = 'None'

#define event colours
typhoon = '#0033cc' # dark blue
earthquake = '#00c5cc' # aqua blue
cyclone = '#cc5100' # orange
plane_crash = '#efef13' # yellow
hurricane = '#55ef13' # light green
volcano_eruption = '#0a5b1f' # dark green
flood = '#ef09d4' # pink
school_attack = '#0e000f' # black

# create the plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, facecolor='#ffffff', frame_on=False)
ax.set_title('Central Location of Identified Disaters (NB)', fontsize=24, color='#333333')

# draw the basemap and its features

m = Basemap(projection='kav7', lon_0=0, resolution='l', area_thresh=10000)
m.drawmapboundary(color=border_color, fill_color=water_color)
m.drawcoastlines(color=coastline_color)
m.drawcountries(color=border_color)
m.fillcontinents(color=land_color, lake_color=water_color)
m.drawparallels(np.arange(-90., 120., 30.), color=meridian_color)
m.drawmeridians(np.arange(0., 420., 60.), color=meridian_color)

#setting the legend for the plot using patch
ty = mpatches.Patch(color = typhoon, label= "Typhoon")
ea = mpatches.Patch(color = earthquake, label = "Earthquake")
cy = mpatches.Patch(color = cyclone, label= "Cyclone")
pl = mpatches.Patch(color = plane_crash, label= "Plane Crash")
hu = mpatches.Patch(color = hurricane, label= "Hurricane")
vo = mpatches.Patch(color = volcano_eruption, label= "Volcano Eruption")
fl = mpatches.Patch(color = flood, label= "Flood")
sc = mpatches.Patch(color = school_attack, label= "School Attack")

plt.legend(handles=[ty,ea,cy,pl,hu,vo,fl,sc],title='Colour of disasters', loc=3)

# project our points from each dataset then concatenate and scatter plot them

# earthquake
try:
    for index, row in rep_points_earth_nb.iterrows():
        rep_points_earth_nb['event_type'] = 'earthquake'
        x_earth_nb,y_earth_nb = m(rep_points_earth_nb['lon'].values, rep_points_earth_nb['lat'].values)
        m.scatter(x_earth_nb, y_earth_nb, s=10, color=earthquake, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass

# typhoon
try:
    for index, row in rep_points_typh_nb.iterrows():
        rep_points_typh_nb['event_type'] = 'typhoon'
        x_typh_nb,y_typh_nb = m(rep_points_typh_nb['lon'].values, rep_points_typh_nb['lat'].values)
        m.scatter(x_typh_nb, y_typh_nb, s=10, color=typhoon, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
  
# cyclone  
try:
    for index, row in rep_points_cyclone_nb.iterrows():
        rep_points_cyclone_nb['event_type'] = 'cyclone'
        x_cyclone_nb,y_cyclone_nb = m(rep_points_cyclone_nb['lon'].values, rep_points_cyclone_nb['lat'].values)
        m.scatter(x_cyclone_nb, y_cyclone_nb, s=10, color=cyclone, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# plane_crash 
try:
    for index, row in rep_points_plane_crash_nb.iterrows():
        rep_points_plane_crash_nb['event_type'] = 'plane_crash'
        x_plane_crash_nb,y_plane_crash_nb = m(rep_points_plane_crash_nb['lon'].values, rep_points_plane_crash_nb['lat'].values)
        m.scatter(x_plane_crash_nb, y_plane_crash_nb, s=10, color=plane_crash, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# hurricane   
try:
    for index, row in rep_points_hurricane_nb.iterrows():
        rep_points_hurricane_nb['event_type'] = 'hurricane'
        x_hurricane_nb,y_hurricane_nb = m(rep_points_hurricane_nb['lon'].values, rep_points_hurricane_nb['lat'].values)
        m.scatter(x_hurricane_nb, y_hurricane_nb, s=10, color=hurricane, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# volcano 
try:
    for index, row in rep_points_volcano_eruption_nb.iterrows():
        rep_points_volcano_eruption_nb['event_type'] = 'volcano_eruption'
        x_volcano_eruption_nb,y_volcano_eruption_nb = m(rep_points_volcano_eruption_nb['lon'].values, rep_points_volcano_eruption_nb['lat'].values)
        m.scatter(x_volcano_eruption_nb, y_volcano_eruption_nb, s=10, color=volcano_eruption, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# flood  
try:
    for index, row in rep_points_flood_nb.iterrows():
        rep_points_flood_nb['event_type'] = 'flood'
        x_flood_nb,y_flood_nb = m(rep_points_flood_nb['lon'].values, rep_points_flood_nb['lat'].values)
        m.scatter(x_flood_nb, y_flood_nb, s=10, color=flood, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass
    
# school_attack 
try:
    for index, row in rep_points_school_attack_nb.iterrows():
        rep_points_school_attack_nb['event_type'] = 'school_attack'
        x_school_attack_nb,y_school_attack_nb = m(rep_points_school_attack_nb['lon'].values, rep_points_school_attack_nb['lat'].values)
        m.scatter(x_school_attack_nb, y_school_attack_nb, s=10, color=school_attack, edgecolor=marker_edge_color, 
                  alpha=1, zorder=3)
except:
    pass

# show the map and save the map
plotname = "nb_cluster_plot_" + points + ".png"
plt.savefig(plotname, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


