import logging as log
import os
import pandas as pd
import simplejson as json
import sys

from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool
from pandas.io.json import json_normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from svm import TweetSVC
from tbparser import TweeboParser
from util import log_mcall

TWEETS_ROOT = 'data/bullyingV3'
TWEETS_FILENAME = f'{TWEETS_ROOT}/tweet.json'
LABELS_FILENAME = f'{TWEETS_ROOT}/data.csv'

TBPARSER_ROOT = 'deps/TweeboParser'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print debug information",
        action='store_const', dest='log_level', const=log.DEBUG,
        default=log.WARNING
    )
    parser.add_argument(
        '-m', '--max-tweets',
        help="Maximum number of tweets to load",
        dest='max_tweets', type=int,
        default=-1
    )
    return parser.parse_args()

def load_tweets(max_tweets=-1):
    log_mcall()
    with open(TWEETS_FILENAME) as tweets_file:
        tweets = json.load(tweets_file)
    
    for tweet in tweets:
        if 'entities' in tweet:
            del tweet['entities']
        if 'user' in tweet:
            del tweet['user']
    X = json_normalize(tweets)

    X.drop('id', axis=1, inplace=True)
    X.rename(columns={'id_str': 'id'}, inplace=True)
    X.drop_duplicates('id', inplace=True)
    X.set_index('id', inplace=True)

    if max_tweets != -1:
        X = X.head(n=max_tweets)

    return X[['text']]

def load_tweet_labels(X):
    log_mcall()
    Y = pd.read_csv(LABELS_FILENAME,
                    names=['id', 'user_id', 'is_trace', 'type', 'form', 'teasing', 'author_role', 'emotion'],
                    dtype={'id': object, 'user_id': object})
    Y.drop_duplicates('id', inplace=True)
    Y.set_index('id', inplace=True)
    Y['is_trace'] = Y['is_trace'] == 'y'

    # Drop labels for entries that are missing in X
    # (X and Y may not have the same number of rows as Twitter has removed some tweets)
    X['tweet_absent'] = False
    X_Y = pd.concat([X, Y], axis=1)
    X_Y.drop(X_Y.index[X_Y['tweet_absent'].isna()], axis=0, inplace=True)

    Y = X_Y.drop(columns=X.columns.values)
    return Y[['is_trace']]

def parse_tweets(X, tbparser_root, tweets_filename='tweets.txt'):
    log_mcall()
    tweets = sorted(X['text'])
    parser = TweeboParser(tbparser_root=tbparser_root,
                          tweets_filename=tweets_filename)
    return parser.parse_tweets(tweets)

def add_tweet_index(X):
    log_mcall()
    tweets = sorted(X['text'])
    index_map = {tweet: index for index, tweet in enumerate(tweets)}
    X['tweet_index'] = X['text'].apply(lambda tweet: index_map[tweet])
    return X

def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    X = load_tweets(max_tweets=args.max_tweets)
    Y = load_tweet_labels(X)
    assert X.shape[0] == Y.shape[0]

    # Use CMU's TweeboParser to produce a dependency tree for each tweet.
    trees = parse_tweets(X, tbparser_root=TBPARSER_ROOT)
    X = add_tweet_index(X)
    X.drop('text', axis=1, inplace=True)

    # NLP Task A: Bullying trace classification
    y = Y['is_trace']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for kernel in 'ptk', 'sptk', 'csptk':
        svc = TweetSVC(trees=trees, tree_kernel=kernel)
        svc.fit(X_train, y_train)
        y_predict = svc.predict(X_test)

        score = accuracy_score(y_true=y_test, y_pred=y_predict)
        print(f"Task A, {kernel} score: {score}")

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    seconds = (end - start).seconds
    print(f"Finished running in {seconds}s", file=sys.stderr)
