import simplejson as json

from pandas.io.json import json_normalize

def main():
    with open('data/bullyingV3/tweet.json') as tweet_file:
        tweet_data = json.load(tweet_file)
    
    for tweet in tweet_data:
        if 'entities' in tweet:
            del tweet['entities']
        if 'user' in tweet:
            del tweet['user']

    df = json_normalize(tweet_data)
    print(df.head(n=10))
    print(df.columns.values)

if __name__ == '__main__':
    main()
