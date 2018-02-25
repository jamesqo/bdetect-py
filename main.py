import pandas as pd
import simplejson as json

from pandas.io.json import json_normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from TreeKernelSVC import TreeKernelSVC

TWEETS_ROOT = 'data/bullyingV3'
TWEETS_FILENAME = f'{TWEETS_ROOT}/tweet.json'
LABELS_FILENAME = f'{TWEETS_ROOT}/data.csv'

def load_tweets():
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
    return X

def f(vals):
    labels, shape = algorithms.factorize(
        vals, size_hint=min(len(self), _SIZE_HINT_LIMIT))
    return labels.astype('i8', copy=False), len(shape)

def load_tweet_labels(X):
    y = pd.read_csv(LABELS_FILENAME,
                    names=['id', 'user_id', 'is_trace', 'type', 'form', 'teasing', 'author_role', 'emotion'],
                    dtype={'id': object, 'user_id': object})
    y.drop_duplicates('id', inplace=True)
    y.set_index('id', inplace=True)
    y['is_trace'] = y['is_trace'] == 'y'

    # Drop labels for entries that are missing in X
    # (X, y may not have the same number of rows as Twitter has removed some tweets)
    X['tweet_absent'] = False
    X_y = pd.concat([X, y], axis=1)
    X_y.drop(X_y.index[X_y['tweet_absent'].isna()], axis=0, inplace=True)

    y = X_y.drop(columns=X.columns.values)
    return y

def main():
    X = load_tweets()
    y = load_tweet_labels(X)
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for kernel in 'ptk', 'sptk', 'csptk':
        tree_clf = TreeKernelSVC(kernel=kernel)
        tree_clf.fit(X_train, y_train)
        y_predict = tree_clf.predict(X_test)

        score = accuracy_score(y_true=y_test, y_pred=y_predict)
        print(f"{kernel} score: {score}")

if __name__ == '__main__':
    main()
