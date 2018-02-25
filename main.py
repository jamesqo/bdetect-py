import pandas as pd
import simplejson as json

from nltk.tree import Tree
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from spacy.lang.en import English

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

def tok_format(tok):
    #print(tok.orth_)
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node):
    return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])

def main():
    X = load_tweets()
    y = load_tweet_labels(X)
    assert X.shape[0] == y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp("The quick brown fox jumps over the lazy dog.")
    for sent in doc.sents:
        print(list(sent.root.children))
        to_nltk_tree(sent.root).pretty_print()

if __name__ == '__main__':
    main()
