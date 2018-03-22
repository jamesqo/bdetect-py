import html
import pandas as pd
import simplejson as json

from pandas.io.json import json_normalize

from util import log_call

def load_tweets(tweets_fname, max_tweets=-1):
    log_call()
    with open(tweets_fname, encoding='utf-8') as tweets_file:
        tweets = json.load(tweets_file)
    
    for tweet in tweets:
        if 'entities' in tweet:
            del tweet['entities']
        if 'user' in tweet:
            del tweet['user']

    X = json_normalize(tweets)

    X.drop(columns=['id'], inplace=True)
    X.rename(columns={'id_str': 'id'}, inplace=True)
    X.drop_duplicates('id', inplace=True)
    X.set_index('id', inplace=True)

    X.rename(columns={'text': 'tweet'}, inplace=True)
    X['tweet'] = X['tweet'].apply(html.unescape)
    X.sort_values('tweet', inplace=True)
    
    if max_tweets != -1:
        X = X.head(n=max_tweets)

    return X[['tweet']]

def load_tweet_labels(labels_fname, X):
    log_call()
    Y = pd.read_csv(labels_fname,
                    names=['id', 'user_id', 'is_trace', 'type', 'form', 'teasing', 'author_role', 'emotion'],
                    dtype={'id': object, 'user_id': object})
    Y.drop_duplicates('id', inplace=True)
    Y.set_index('id', inplace=True)
    Y['is_trace'] = Y['is_trace'] == 'y'

    # Drop labels for entries that are missing in X
    # (X and Y may not have the same number of rows as Twitter has removed some tweets)
    X['tweet_absent'] = False
    X_Y = pd.concat([X, Y], axis=1)
    X.drop(columns=['tweet_absent'], inplace=True)
    X_Y.drop(index=X_Y.index[X_Y['tweet_absent'].isna()], inplace=True)

    Y = X_Y.drop(columns=X.columns.values)
    return Y[['is_trace']]

def add_tweet_index(X):
    log_call()
    tweets = sorted(X['tweet'])
    index_map = {tweet: index for index, tweet in enumerate(tweets)}
    X['tweet_index'] = X['tweet'].apply(lambda tweet: index_map[tweet])
    return X
