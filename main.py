import logging as log
import nltk
import os
import pandas as pd
import sys

from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_loader import add_tweet_index, load_tweets, load_tweet_labels
from svm import TweetSVC
from tbparser import TweeboParser
from util import log_mcall

TWEETS_ROOT = os.path.join('data', 'bullyingV3')
TWEETS_FNAME = os.path.join(TWEETS_ROOT, 'tweet.json')
LABELS_FNAME = os.path.join(TWEETS_ROOT, 'data.csv')

TBPARSER_ROOT = os.path.join('deps', 'TweeboParser')
TBPARSER_INPUT_FNAME = 'tweets.txt'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        metavar='C',
        help="set C hyperparameter of svm",
        dest='c',
        action='store',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        dest='log_level',
        action='store_const',
        const=log.DEBUG,
        default=log.WARNING
    )
    parser.add_argument(
        '-m', '--max-tweets',
        metavar='LIMIT',
        help="load at most LIMIT tweets into the corpus. useful for quick debugging",
        dest='max_tweets',
        action='store',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-n', '--n-jobs',
        metavar='N',
        help="use N cpus to compute gram matrix in parallel. N=-1 means use all cpus, N=-2 means use all but 1 cpu, and so on. setting N=1 is recommended for debugging",
        dest='n_jobs',
        action='store',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-r', '--refresh-predictions',
        help="refresh predictions by re-running TweeboParser on the corpus (this will take a while)",
        dest='refresh_predictions',
        action='store_true'
    )
    return parser.parse_args()

def parse_tweets(X, tbparser_root, tweets_fname, refresh_predictions=False, scrub_trivia=True, lemmatize=True):
    log_mcall()
    tweets = sorted(X['text'])
    parser = TweeboParser(tbparser_root=tbparser_root,
                          tweets_fname=tweets_fname,
                          refresh_predictions=refresh_predictions)
    trees = parser.parse_tweets(tweets)
    if scrub_trivia:
        trees = _scrub_trivia(trees)
    if lemmatize:
        trees = _lemmatize(trees)
    return list(trees)

def _scrub_trivia(trees):
    log_mcall()
    # Filter out nodes with HEAD = -1 from the dependency tree, except for
    # hashtags and @ mentions which provide valuable information.
    # Such nodes are direct children of the root node, so we don't need to
    # exhaustively search the tree.
    NONTRIVIA_TAGS = ['#', '@']
    for tree in trees:
        tree.children[:] = [child for child in tree.children
                                  if child.data['head'] != -1 or
                                     child.data['upostag'] in NONTRIVIA_TAGS]
    return trees

def _lemmatize(trees):
    def do_lemmatize(node):
        assert node.data['lemma'] == '_'

        form = node.data['form']
        lemma = stem.stem(form)
        node.data['lemma'] = lemma

        for child in node.children:
            do_lemmatize(child)

    log_mcall()
    if not nltk.download('wordnet', quiet=True):
        raise RuntimeError("Failed to download WordNet corpus")

    # TODO: Decide experimentally if lemmatization or stemming (or both) performs better.
    stem = PorterStemmer()
    for tree in trees:
        for child in tree.children:
            do_lemmatize(child)

    return trees

def print_scores(task, model, y_test, y_predict):
    scores = OrderedDict([
        ('accuracy', accuracy_score(y_true=y_test, y_pred=y_predict)),
        ('precision', precision_score(y_true=y_test, y_pred=y_predict)),
        ('recall', recall_score(y_true=y_test, y_pred=y_predict)),
        ('f1', f1_score(y_true=y_test, y_pred=y_predict))
    ])

    print("task {}, model {}".format(task, model))
    for name, score in scores.items():
        print("{}: {}".format(name, score))
    print()

def save_test_session(tweets_test, y_test, y_predict):
    with open('test_set.log', 'w', encoding='utf-8') as test_set_file:
        contents = '\n'.join(tweets_test)
        test_set_file.write(contents)
    y_test.to_csv('labels.log', index=False)
    with open('predictions.log', 'w', encoding='utf-8') as predict_file:
        contents = '\n'.join(map(str, y_predict))
        predict_file.write(contents)

def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    X = load_tweets(tweets_fname=TWEETS_FNAME, max_tweets=args.max_tweets)
    Y = load_tweet_labels(labels_fname=LABELS_FNAME, X=X)
    assert X.shape[0] == Y.shape[0]

    # Use CMU's TweeboParser to produce a dependency tree for each tweet.
    trees = parse_tweets(X,
                         tbparser_root=TBPARSER_ROOT,
                         tweets_fname=TBPARSER_INPUT_FNAME,
                         refresh_predictions=args.refresh_predictions)
    assert len(trees) == X.shape[0]

    tweets = X['text']
    X = add_tweet_index(X)
    X.drop('text', axis=1, inplace=True)

    # NLP Task A: Bullying trace classification
    y = Y['is_trace']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for kernel in ['ptk',]: #'sptk', 'csptk']:
        svc = TweetSVC(trees=trees, tree_kernel=kernel, C=args.c)
        svc.fit(X_train, y_train, n_jobs=args.n_jobs)
        y_predict = svc.predict(X_test, n_jobs=args.n_jobs)
        print_scores(task='a', model='svm+{}'.format(kernel), y_test=y_test, y_predict=y_predict)

        tweets_test = [tweets[index] for index in X_test['tweet_index']]
        save_test_session(tweets_test=tweets_test, y_test=y_test, y_predict=y_predict)

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    seconds = (end - start).seconds
    print("Finished running in {}s".format(seconds), file=sys.stderr)
