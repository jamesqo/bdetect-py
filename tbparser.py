import conllu
import os

from collections import namedtuple, OrderedDict
from itertools import islice

from util import exec_and_check, log_mcall

def _add_root(graph, tweet):
    # Turns a multi-rooted graph into a tree by adding a root node.
    return TreeRoot(children=graph, tweet=tweet)

def _remove_newlines(tweet):
    return tweet.replace('\n', ' ').replace('\r', ' ')

class TreeRoot(object):
    def __init__(self, children, tweet):
        self.children = children
        self._tweet = tweet

    def __str__(self):
        return self._tweet

class TweeboParser(object):
    def __init__(self, tbparser_root, tweets_filename, refresh_predictions=False):
        log_mcall()
        tbparser_root = tbparser_root.rstrip('/')
        tbparser_root = os.path.abspath(tbparser_root)
        tweets_filename = os.path.join(tbparser_root, tweets_filename)

        self._tbparser_root = tbparser_root
        self._tweets_filename = tweets_filename
        self._output_filename = f'{tweets_filename}.predict'
        self._run_scripts = refresh_predictions or not os.path.isfile(self._output_filename)

        if self._run_scripts:
            # Run TweeboParser install script
            exec_and_check(f'cd {tbparser_root} && bash install.sh')

    def parse_tweets(self, tweets):
        log_mcall()
        if self._run_scripts:
            with open(self._tweets_filename, 'w', encoding='utf-8') as tweets_file:
                # Twitter permits newlines in tweets, causing problems with the dependency parser
                # which expects one tweet per line in the input file.
                tweets = [_remove_newlines(tweet) for tweet in tweets]
                contents = '\n'.join(tweets)
                tweets_file.write(contents)

            # Run CMU's parser
            # TODO: It hangs for a long time after making the predictions because it generates 4000+
            # files (one per tweet) in working_dir/test_score. Find out how to make it stop that.
            exec_and_check(f'cd {self._tbparser_root} && bash run.sh {self._tweets_filename}')

        # Parse output file, which is formatted in CoNLL-X
        # Since it doesn't use the PHEAD or PDEPREL fields, we can use a CoNLL-U parser library
        with open(self._output_filename, 'r', encoding='utf-8') as output_file:
            contents = output_file.read().strip()
        batches = contents.split('\n\n')
        batches = islice(batches, len(tweets))
        graphs = map(conllu.parse_tree, batches)
        trees = [_add_root(graph, tweet) for graph, tweet in zip(graphs, tweets)]
        return trees
