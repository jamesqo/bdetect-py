import pandas as pd
import simplejson as json
import spacy

from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
#from stat_parser import Parser

def load_tweets():
    with open('data/bullyingV3/tweet.json') as tweets_file:
        tweets = json.load(tweets_file)
    
    for tweet in tweets:
        if 'entities' in tweet:
            del tweet['entities']
        if 'user' in tweet:
            del tweet['user']

    X = json_normalize(tweets)
    return X

def load_tweet_labels(X):
    y = pd.read_csv('data/bullyingV3/data.csv', names=['id', 'user_id', 'is_trace', 'type', 'form', 'teasing', 'author_role', 'emotion'])
    y['is_trace'] = y['is_trace'] == 'y'

    # Drop labels for entries that are missing in X
    # (X, y may not have the same number of rows as Twitter has removed some tweets)
    X['tweet_absent'] = False
    X_y = pd.concat([X, y], axis=1)
    X_y.drop(X_y.index[X_y['tweet_absent'].isna()], axis=0, inplace=True)

    dropcols = list(X.columns.values)
    dropcols.remove('id')
    y = X_y.drop(columns=dropcols)
    return y

def main():
    X = load_tweets()
    y = load_tweet_labels(X)
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nlp = spacy.load('en')
    doc = nlp("Hello, world!")
    print(doc)

if __name__ == '__main__':
    main()
