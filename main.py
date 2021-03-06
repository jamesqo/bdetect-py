import logging as log
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

from data_prep import add_tweet_index, load_tweets, load_tweet_labels
from svm import TreeSVC
from tbparser import TweeboParser
from util import log_call

TWEETS_ROOT = os.path.join('data', 'bullyingV3')
TWEETS_FNAME = os.path.join(TWEETS_ROOT, 'tweet.json')
LABELS_FNAME = os.path.join(TWEETS_ROOT, 'data.csv')

TBPARSER_ROOT = os.path.join('deps', 'TweeboParser')
TBPARSER_INPUT_FNAME = 'tweets.txt'

DEFAULT_LAMBDA = 0.5
DEFAULT_MU = 0.1
DEFAULT_C = 100.0
DEFAULT_ITERATIONS = 100

FIT_SAVEPATH = 'kernels.fit.csv'
PREDICT_SAVEPATH = 'kernels.predict.csv'
TEST_SET_SAVEPATH = 'test_set.log'
LABELS_SAVEPATH = 'labels.log'
PREDICTIONS_SAVEPATH = 'predictions.log'

GRID_SEARCH_IGNORE_WARNINGS = [UserWarning, UndefinedMetricWarning]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        action='store_const',
        dest='log_level',
        const=log.DEBUG,
        default=log.WARNING
    )
    parser.add_argument(
        '-m', '--max-tweets',
        metavar='LIMIT',
        help="load at most LIMIT tweets into the corpus. useful for quick debugging",
        action='store',
        dest='max_tweets',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-n', '--n-jobs',
        metavar='N',
        help="use N cpus to compute gram matrix in parallel. N=-1 means use all cpus, N=-2 means use all but 1 cpu, and so on. setting N=1 is recommended for debugging",
        action='store',
        dest='n_jobs',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-o', '--optimize-params',
        metavar='ITERATIONS',
        help="run randomized search ITERATIONS times to optimize hyperparameters. ITERATIONS defaults to {}".format(DEFAULT_ITERATIONS),
        action='store',
        nargs='?',
        dest='n_iter',
        type=int,
        const=DEFAULT_ITERATIONS,
        default=-1
    )
    parser.add_argument(
        '-r', '--refresh-predictions',
        help="refresh predictions by re-running TweeboParser on the corpus (this will take a while)",
        action='store_true',
        dest='refresh_predictions'
    )
    parser.add_argument(
        '-v', '--visualize',
        help="display visualization of fit() kernel matrix",
        action='store_true',
        dest='visualize'
    )
    return parser.parse_args()

def fit(clf, X_train, y_train, args):
    log_call()

    should_optimize_params = args.n_iter != -1
    if should_optimize_params:
        # NOTE: It would be cleaner to pass an arbitrary function to run on the
        # first fit() call so warning handling wouldn't be coupled to the class.
        # However that proved to be problematic since lambdas can't be pickled.
        clf.ignore_warnings = GRID_SEARCH_IGNORE_WARNINGS
        best_params = optimize_params(estimator=clf,
                                      X_train=X_train,
                                      y_train=y_train,
                                      n_iter=args.n_iter,
                                      n_jobs=args.n_jobs)
        # Reset ignore_warnings so the original copy of clf, which hasn't been
        # fitted yet, doesn't ignore any warnings during fit()
        clf.ignore_warnings = []

        print_best_params(best_params)
        clf.set_params(**best_params)

    clf.fit(X=X_train,
            y=y_train,
            n_jobs=args.n_jobs,
            savepath=FIT_SAVEPATH)

def optimize_params(estimator, X_train, y_train, n_iter, n_jobs):
    log_call()
    param_grid = get_param_grid()
    n_iter = min(n_iter, len(ParameterGrid(param_grid)))
    clf = RandomizedSearchCV(estimator=estimator,
                             param_distributions=param_grid,
                             n_iter=n_iter,
                             scoring='f1',
                             n_jobs=n_jobs,
                             refit=False,
                             random_state=42)
    clf.fit(X_train, y_train)
    return clf.best_params_

def get_param_grid():
    return {
        'estimator__C': 10 ** np.arange(-2, 3, dtype=np.float64),
        'lambda_': np.linspace(0.5, 1, 6),
        'mu': np.linspace(0.1, 0.5, 5)
    }

def visualize(kmat, labels):
    log_call()
    m = labels.shape[0]
    assert kmat.shape == (m, m)

    kpca = KernelPCA(n_components=3,
                     kernel='precomputed',
                     random_state=42)
    kmat_reduced = kpca.fit_transform(kmat)

    x, y, z = kmat_reduced[:, 0], kmat_reduced[:, 1], kmat_reduced[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=labels, cmap='cool')
    plt.show()

def predict(clf, X_test, **kwargs):
    log_call()
    return clf.predict(X_test, **kwargs)

def print_header(task, model):
    print("task {}, model {}".format(task, model))
    print()

def print_best_params(best_params):
    print("best hyperparameter values:")
    for key in sorted(best_params.keys()):
        print("{}: {}".format(key, best_params[key]))
    print()

def print_scores(y_test, y_predict):
    values = OrderedDict([
        ('true instances', "actual {}, predicted {}".format(sum(y_test), sum(y_predict))),
        ('false instances', "actual {}, predicted {}".format(sum(~y_test), sum(~y_predict))),
        ('accuracy', accuracy_score(y_true=y_test, y_pred=y_predict)),
        ('precision', precision_score(y_true=y_test, y_pred=y_predict)),
        ('recall', recall_score(y_true=y_test, y_pred=y_predict)),
        ('f1', f1_score(y_true=y_test, y_pred=y_predict))
    ])

    for name, value in values.items():
        print("{}: {}".format(name, value))

    print("confusion matrix:")
    print(confusion_matrix(y_true=y_test, y_pred=y_predict))

    print()

def save_test_session(tweets_test, y_test, y_predict):
    with open(TEST_SET_SAVEPATH, 'w', encoding='utf-8') as test_set_file:
        contents = '\n'.join(tweets_test) + '\n'
        test_set_file.write(contents)
    y_test.to_csv(LABELS_SAVEPATH, index=False)
    with open(PREDICTIONS_SAVEPATH, 'w', encoding='utf-8') as predict_file:
        contents = '\n'.join(map(str, y_predict)) + '\n'
        predict_file.write(contents)

def task_a(X, Y, tweets, trees, args):
    log_call()
    y = Y['is_trace']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for kernel in ['ptk']: # 'sptk', 'csptk'
        print_header(task='a', model='svm+{}'.format(kernel))

        base_clf = SVC(C=DEFAULT_C,
                       class_weight='balanced')
        clf = TreeSVC(estimator=base_clf,
                      kernel=kernel,
                      trees=trees,
                      lambda_=DEFAULT_LAMBDA,
                      mu=DEFAULT_MU)
        fit(clf, X_train, y_train, args)
        if args.visualize:
            visualize(kmat=clf.kernel_matrix_, labels=y_train)

        y_predict = predict(clf, X_test, n_jobs=args.n_jobs, savepath=PREDICT_SAVEPATH)
        print_scores(y_test=y_test, y_predict=y_predict)

        tweets_test = [tweets[index] for index in X_test['tweet_index']]
        save_test_session(tweets_test=tweets_test, y_test=y_test, y_predict=y_predict)

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
    task_a(X, Y, tweets, trees, args)

if __name__ == '__main__':
    start = datetime.now()
    main()
    end = datetime.now()
    seconds = (end - start).seconds
    log.debug("finished running in {}s".format(seconds))
