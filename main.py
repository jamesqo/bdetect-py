import logging as log
import os
import sys

from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from data_prep import add_tweet_index, load_tweets, load_tweet_labels
from svm import TweetSVC
from tbparser import TweeboParser
from util import log_call

TWEETS_ROOT = os.path.join('data', 'bullyingV3')
TWEETS_FNAME = os.path.join(TWEETS_ROOT, 'tweet.json')
LABELS_FNAME = os.path.join(TWEETS_ROOT, 'data.csv')

TBPARSER_ROOT = os.path.join('deps', 'TweeboParser')
TBPARSER_INPUT_FNAME = 'tweets.txt'

FIT_SAVEPATH = 'kernels.fit.csv'
PREDICT_SAVEPATH = 'kernels.predict.csv'
TEST_SET_SAVEPATH = 'test_set.log'
LABELS_SAVEPATH = 'labels.log'
PREDICTIONS_SAVEPATH = 'predictions.log'

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
    with open(TEST_SET_SAVEPATH, 'w', encoding='utf-8') as test_set_file:
        contents = '\n'.join(tweets_test) + '\n'
        test_set_file.write(contents)
    y_test.to_csv(LABELS_SAVEPATH, index=False)
    with open(PREDICTIONS_SAVEPATH, 'w', encoding='utf-8') as predict_file:
        contents = '\n'.join(map(str, y_predict)) + '\n'
        predict_file.write(contents)

def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    X = load_tweets(TWEETS_FNAME, max_tweets=args.max_tweets)
    Y = load_tweet_labels(LABELS_FNAME, X)
    assert X.shape[0] == Y.shape[0]

    # Use CMU's TweeboParser to produce a dependency tree for each tweet.
    tweets = sorted(X['tweet'])
    parser = TweeboParser(tbparser_root=TBPARSER_ROOT,
                          input_fname=TBPARSER_INPUT_FNAME,
                          refresh_predictions=args.refresh_predictions)
    trees = parser.parse_tweets(tweets)
    assert len(trees) == X.shape[0]

    X = add_tweet_index(X)
    X.drop(columns=['tweet'], inplace=True)

    # NLP Task A: Bullying trace classification
    y = Y['is_trace']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for kernel in ['ptk']: # 'sptk', 'csptk'
        svc = TweetSVC(trees=trees, tree_kernel=kernel, C=args.c)
        svc.fit(X_train, y_train, savepath=FIT_SAVEPATH, n_jobs=args.n_jobs)
        y_predict = svc.predict(X_test, savepath=PREDICT_SAVEPATH, n_jobs=args.n_jobs)
        print_scores(task='a', model='svm+{}'.format(kernel), y_test=y_test, y_predict=y_predict)

        tweets_test = [tweets[index] for index in X_test['tweet_index']]
        save_test_session(tweets_test=tweets_test, y_test=y_test, y_predict=y_predict)

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    seconds = (end - start).seconds
    print("Finished running in {}s".format(seconds), file=sys.stderr)
