import conllu
import os

from collections import namedtuple, OrderedDict
from util import exec_and_check

TreeNode = namedtuple('TreeNode', ['data', 'children'])

def _add_root(graph):
    # Turns a multi-rooted graph into a tree by adding a root node.
    data = OrderedDict([('id', 0)])
    return TreeNode(data=data, children=graph)

def _remove_newlines(tweet):
    return tweet.replace('\n', ' ').replace('\r', ' ')

class TweeboParser(object):
    def __init__(self, tbparser_root, tweets_filename, refresh_predictions=False):
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
        if self._run_scripts:
            with open(self._tweets_filename, 'w', encoding='utf-8') as tweets_file:
                # Twitter permits newlines in tweets, causing problems with the dependency parser
                # which expects one tweet per line in the input file.
                tweets = map(_remove_newlines, tweets)
                contents = '\n'.join(tweets)
                tweets_file.write(contents)

            # Run CMU's parser
            exec_and_check(f'cd {self._tbparser_root} && bash run.sh {self._tweets_filename}')

        # Parse output file, which is formatted in CoNLL-X
        # Since it doesn't use the PHEAD or PDEPREL fields, we can use a CoNLL-U parser library
        with open(self._output_filename, 'r', encoding='utf-8') as output_file:
            contents = output_file.read().strip()
        batches = contents.split('\n\n')
        # TODO: conllu.parse_tree is ignoring tokens with HEAD = -1 like hashtags, @ mentions, URLs, etc.
        graphs = map(conllu.parse_tree, batches)
        trees = map(_add_root, graphs)
        return trees
