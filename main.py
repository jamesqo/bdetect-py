import pandas as pd
import simplejson as json

from pandas.io.json import json_normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from TreeKernelClassifier import TreeKernelClassifier

TWEETS_ROOT = 'data/bullyingV3'
CORENLP_ROOT = 'lib/stanford-corenlp-full-2018-01-31'
CORENLP_VER = '3.9.0'

def load_tweets():
    with open(f'{TWEETS_ROOT}/tweet.json') as tweets_file:
        tweets = json.load(tweets_file)
    
    for tweet in tweets:
        if 'entities' in tweet:
            del tweet['entities']
        if 'user' in tweet:
            del tweet['user']

    X = json_normalize(tweets)
    return X

def load_tweet_labels(X):
    y = pd.read_csv(f'{TWEETS_ROOT}/data.csv',
                    names=['id', 'user_id', 'is_trace', 'type', 'form', 'teasing', 'author_role', 'emotion'])
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

def get_jar_paths():
    jar_path = f'{CORENLP_ROOT}/stanford-corenlp-{CORENLP_VER}.jar'
    models_jar_path = f'{CORENLP_ROOT}/stanford-corenlp-{CORENLP_VER}-models.jar'
    return jar_path, models_jar_path

def main():
    X = load_tweets()
    y = load_tweet_labels(X)
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree_clf = TreeKernelClassifier(*get_jar_paths(), kernel='ptk')
    tree_clf.fit(X_train, y_train)
    y_predict = tree_clf.predict(X_test)

    score = accuracy_score(y_true=y_test, y_pred=y_predict)
    print(f"Tree kernel score: {score}")

if __name__ == '__main__':
    main()
