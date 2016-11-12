''' It's the very first version of tweeter data mining app
    Here I just attemepted to find special hashtags and extract some other related data to that specific hashtag and also the
    user which already used that. 
    Mojtaba Fazli Oct, 28, 2016
    Using this code by mentioning the code developer and details is allowed.
'''
#***********************************************************************************************************
#* Importing needed libraries 
#***********************************************************************************************************
import tweepy
import pandas as pd
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from time import sleep
import csv
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument ("-q", required = True, 
	help = "the hashtag label accompanied by #")
parser.add_argument ("-i", required = True, 
	help = "the last ith tweets you need ")
parser.add_argument ("-f", required = True, 
	help = "a name for the CSV file")
args = vars(parser.parse_args())


#***********************************************************************************************************
#* setting the panda variables ( however here we are not using it , but it's good for next versions) 
#***********************************************************************************************************

pd.options.display.max_columns = 60
pd.options.display.max_rows= 60
pd.options.display.width= 120

#***********************************************************************************************************
#* Setting the authentication and access key details
#***********************************************************************************************************

access_token="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

#***********************************************************************************************************
#* setting the key parameters including the hashtag we're tending to search, number of items to be extracted 
#* and the FileName in which we're going to save our data as CSV file. 
#***********************************************************************************************************

hashtag = str('#' + str(args['q'])) ##VitalSigns'
No_of_items = int(args['i']) #200
FileName = str(args['f']) #'tweet123.csv'

print ( " The Program is already connected to twitter website ...")
print (" It's already extracting the last " + str(args['i']) +" tweets which includes <" + str(hashtag) + ">....")

#***********************************************************************************************************
#* Defining a function for extracting the "tweet data" and " User data" in a format asked by Dr. Zion
#***********************************************************************************************************

with open('current_userids.txt', 'r') as f:
    userids = f.readlines()



def DataStrFilling(userids):

    id_list = [tweet.author.id for userid in userids]
    TweetDataSet = pd.DataFrame(id_list, columns=["id"])

    # Processing Tweet Data
    #import sys
    #print(dir(query_results[0]))
    #print(query_results[0]._json['retweeted_status']['id'])
    #for tweet in query_results:
    #	print(dir(tweet.author))

    #		print(dir(tweet))
    #		break

    #sys.exit(0)
    TweetDataSet["CreatedAt"] = [tweet.created_at for tweet in query_results]
    TweetDataSet["Country"] = [tweet.place.country if tweet.place else "None"  for tweet in query_results]
    TweetDataSet["Country_Code"] = [tweet.place.country_code if tweet.place else "None" for tweet in query_results]
    TweetDataSet["Place"] = [tweet.place.name if tweet.place else "None" for tweet in query_results]
    TweetDataSet["RetweetedStatus"] = [tweet.retweeted_status.id_str if hasattr(tweet,'retweeted_status') else 0  for tweet in query_results]
    TweetDataSet["UsrFavouriteCount"] = [tweet.author.favourites_count for tweet in query_results]
    TweetDataSet["Text"] = [tweet.text.replace('\n',' ').replace('\r','') for tweet in query_results]
    TweetDataSet["Lang"] = [tweet.lang for tweet in query_results]
    TweetDataSet["UsrName"] = [tweet.author.name for tweet in query_results]
    TweetDataSet["UserId"] = [tweet.author.id for tweet in query_results]
    TweetDataSet["UserLang"] = [tweet.author.lang for tweet in query_results]
    TweetDataSet["UsrLocation"] = [tweet.author.location if tweet.author.location else "N/A" for tweet in query_results]
    TweetDataSet["UsrFollowersCount"] = [tweet.author.followers_count for tweet in query_results]
    TweetDataSet["UsrFriendsCount"] = [tweet.author.friends_count for tweet in query_results]
    TweetDataSet["UsrVerified"] = [tweet.author.verified for tweet in query_results]
    TweetDataSet["UsrCreatedAt"] = [tweet.author.created_at for tweet in query_results]
    TweetDataSet["UsrDescription"] = [tweet.author.description for tweet in query_results]
    TweetDataSet["UsrGeoEnabled"] = [tweet.author.geo_enabled for tweet in query_results]
    TweetDataSet["UsrStatuesCount"] = [tweet.author.statuses_count for tweet in query_results]
    TweetDataSet["UsrTimeZone"] = [tweet.author.time_zone if tweet.author.time_zone else "N/A" for tweet in query_results]
    TweetDataSet["UsrUtcOffset"] = [tweet.author.utc_offset if tweet.author.utc_offset else "N/A"for tweet in query_results]
    TweetDataSet["Retweeted"] = [tweet.retweeted for tweet in query_results]
    TweetDataSet["Retweet_Cont"] = [tweet.retweet_count for tweet in query_results]
    
    # following 3 items were not mentioned in sample csv file by Dr. Zion but just added by me , seems that sometimes they may be valuable on analysis
   
    TweetDataSet["usrScrName"] = [tweet.author.screen_name for tweet in query_results]
    TweetDataSet["favorite_count"] = [tweet.favorite_count for tweet in query_results]
    TweetDataSet["Source"] = [tweet.source for tweet in query_results]
    
        
    return TweetDataSet
#***********************************************************************************************************
#* authentication steps for connectng to Twitter API
#***********************************************************************************************************

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth.secure = True

api = tweepy.API(auth) # I'm using tweepy here as Twitter API

#***********************************************************************************************************
#* Querying the Favoirte hashtag and appending the results to our query result list 
#***********************************************************************************************************

query_results = []
for tweet in tweepy.Cursor(api.search, q= hashtag).items(No_of_items) : 
    try : 
        query_results.append(tweet)

        #user=tweet.author

        #for param in dir(user):
        #    if not param.startswith("_"):
        #        print "%s : %s" % (param, eval("user." + param))
        #print ("\n\nFound tweets from : @" + tweet.user.screen_name)
        #print ( tweet.entities, "Created_at" + str(tweet.created_at), "Country : " + str(tweet.place))
    except tweepy.TweepError as e:
        print (e.reason)
        sleep(10)
        continue
    except StopIteration :
        break
TweetDataSet = DataStrFilling(userids)
#***********************************************************************************************************
#* Saving the panda data structure into a CSV file
#***********************************************************************************************************

#with open(FileName,'a') as fileHandle:
TweetDataSet.to_csv(FileName, sep = ',', encoding = 'utf-8',header=True)
print ("***************************************************************************************************")
print ("the total data of last " + str(len(query_results)) + " tweets about ( " + str(hashtag) + " ) is already placed in <" + str(FileName) + "> File.")
print ("***************************************************************************************************")
