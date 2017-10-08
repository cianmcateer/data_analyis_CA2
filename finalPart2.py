from __future__ import division
from collections import Counter
import math, random, csv, json
import numpy as np
import requests
import io
import matplotlib.pyplot as plt

from collections import Counter
import ast
##
#
# APIs
#
##

# Twitter
#
####

from twython import Twython
from twython import TwythonStreamer

# fill these in if you want to use the code
CONSUMER_KEY = "zzk9XEggIUJ6hquG1UrOzGDTi"
CONSUMER_SECRET = "vgiyiCg2nDRw3cf7kTP9Gmaflv3ydh61h9x05I6mQuWhTcBTqM"
ACCESS_TOKEN = "750364550-npORFNUeAX8SDVJyoSBma44D7GyB4iYhkZBObfRL"
ACCESS_TOKEN_SECRET = "dOOeVOOhkC2BiDknM9Q6GXlghlnCH9Cf9GCQiv9dAWs3m"

print("TWYTHON ANALYSIS")
users = []
messages = []
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)
def call_twitter_search_api():



    # General search for tweets containing the phrase "brexit"
    for status in twitter.search(q='"brexit"')["statuses"]:
        user = status["user"]["screen_name"].encode('utf-8')
        text = status["text"].encode('utf-8')
        users.append(user)
        messages.append(text)
        print(user, ":", text)
        print()


call_twitter_search_api()
tweets = []

class MyStreamer(TwythonStreamer):
#    """our own subclass of TwythonStreamer that specifies
#    how to interact with the stream"""

    def on_success(self, data):



        # only want to collect English-language tweets
        if data['lang'] == 'en':
            tweets.append(data)
            print("received tweet #", len(tweets))
        top_hashtags = Counter(hashtag['text'].lower()
            for tweet in tweets
            for hashtag in tweet['entities']['hashtags'])
        #collects most common hashtags
        #print(top_hashtags.most_common(10))
        # stop when we've collected enough
        #Save to text file
        saveFile = open('twitterData.txt','a')
        saveFile.write(str(data))
        saveFile.write('\n')
        saveFile.close()
        if len(tweets) >= 15:
             self.disconnect()
        return tweets


    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()

def call_twitter_streaming_api():
    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
                        ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # starts consuming public statuses that contain the keyword 'data'
    stream.statuses.filter(track='data')

#_______________________________________________________________________________
#
#   Reading from file & converting to a dictionary
#
#_______________________________________________________________________________

ftweets = []
tweets_mine = []

def read_ftweets():
    counter = 0
    with io.open('twitterData.txt', 'r' , encoding = 'ISO-8859-1') as f:
        for line in f:
            counter = counter +1
            ftweets.append((line))

def convert_ftweets_to_dict(count):
	for i in range(0, count):
		tweets_mine.append(ast.literal_eval(ftweets[i]))


#call_twitter_streaming_api()
#read_ftweets()
#count=len(ftweets)

#convert_ftweets_to_dict(count)
#print(ftweets)
#_________________________________________________________________________________
#
#   Counting hashtags
#

top_file_hashtags = Counter(hashtag['text'].lower() for tweet_mine in tweets_mine for hashtag in tweet_mine["entities"]["hashtags"])
#print(top_file_hashtags.most_common(5))

"""
Calculate mean length of entered List and draws line graph of list
"""
def getMeanStringLength(x):
    lengths = []
    for v in x:
        lengths.append(len(v))

    plt.plot(lengths)
    plt.show()
    return np.mean(lengths)

meanUserNameLength = float(str(round(getMeanStringLength(users),2)))
print("mean username length",meanUserNameLength)
meanTweetLength = float(str(round(getMeanStringLength(messages),2)))
print("mean tweet length",meanTweetLength)




dundalkGeoCode = '53.9979,6.4060,100km'
bostonGeoCode = '52.9789,0.0266,100km'

def results(query,locale):
        results = twitter.search(q=query,count=20,lang = 'en',include_entities = True,geocode = locale)
        return results['statuses']

def printTweets(results):
    for tweet in results:
        print(tweet['text'])
dundalkResults = results("#brexit",dundalkGeoCode)
bostonResults = results("#brexit",bostonGeoCode)
print("100 KILOMETER RADIUS OF DUNDALK")
printTweets(dundalkResults)
print("")
print("")
print("")
print("100 KILOMETER RADIUS OF BOSTON,LINCONSHIRE")
printTweets(bostonResults)

LA_geocode = "34.0522,118.2437,100mi"
knoxVilleGeocode = "35.9606,83.9207,350mi"

DT_LA_Tweets = results("trump",LA_geocode)
DT_knoxville_tweets = results("trump",knoxVilleGeocode)

print("")
print("100 MILE RADIUS OF LA")
printTweets(DT_LA_Tweets)
print("")
print("")
print("")
print("350 MILE RADIUS OF KNOXVILLE TENNESSEE")
printTweets(DT_knoxville_tweets)
