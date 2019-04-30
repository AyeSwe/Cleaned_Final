import datetime
import pandas as pd
import json

tweets_data_path = 'tweet_data_testing.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")

# reading line by line from json file ( Remember Json are in dictionary format)
for line in tweets_file:
    try:
        tweet = json.loads(line)

        # changing Twitter time format to python yy,m,d

        tweet_daytime = datetime.datetime.fromtimestamp(int(tweet['timestamp_ms']) / 1000)

        tweet_day = tweet_daytime.strftime('%Y-%m-%d')

        #print(tweet_day)
        # appendign to the tweets_data array
        tweets_data.append(tweet)
    except:
        continue

tweets = pd.DataFrame(tweets_data)

# this replace the Date Column of the tweets with converted Date format
tweets['Date'] = tweet_day

print(" tweets['Date']:is  ", tweets['Date'])
print(tweets.info())

#print(tweets)





