#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
import Twitter_credential
from tweepy import OAuthHandler
from tweepy import Stream



#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(Twitter_credential.CONSUMER_KEY, Twitter_credential.CONSUMER_SECRET)
    auth.set_access_token(Twitter_credential.ACCESS_TOKEN, Twitter_credential.ACCESS_TOKEN_SECRET)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(track=['bitcoin','currency'])
# this streamming is captured from command line redirection to the "tweet_data_testing.txt file "