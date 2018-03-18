import conllu
import os

from collections import namedtuple, OrderedDict
from util import exec_and_check

TreeNode = namedtuple('TreeNode', ['data', 'children'])

def _add_root(graph):
    # Turns a multi-rooted graph into a tree by adding a root node.
    data = OrderedDict([('id', 0)])
    return TreeNode(data=data, children=graph)

class TweeboParser(object):
    def __init__(self, tbparser_root, tweets_filename, run_scripts=True):
        tbparser_root = tbparser_root.rstrip('/')
        tbparser_root = os.path.abspath(tbparser_root)
        tweets_filename = os.path.join(tbparser_root, tweets_filename)

        self._tbparser_root = tbparser_root
        self._tweets_filename = tweets_filename
        self._run_scripts = run_scripts

        if run_scripts:
            # Run TweeboParser install script
            install_sh = os.path.join(tbparser_root, 'install.sh')
            exec_and_check(f'bash {install_sh}')

    def parse_tweets(self, tweets):
        if self._run_scripts:
            with open(self._tweets_filename, 'w', encoding='utf-8') as tweets_file:
                contents = '\n'.join(tweets)
                tweets_file.write(contents)

            # Run CMU's parser
            run_sh = os.path.join(self._tbparser_root, 'run.sh')
            exec_and_check(f'bash {run_sh} {self._tweets_filename}')

        # Parse output file, which is formatted in CoNLL-X
        # Since it doesn't use the PHEAD or PDEPREL fields, we can use a CoNLL-U parser library
        output_filename = f'{self._tweets_filename}.predict'
        with open(output_filename, 'r', encoding='utf-8') as output_file:
            contents = output_file.read().strip()
        batches = contents.split('\n\n')
        # TODO: conllu.parse_tree is ignoring tokens with HEAD = -1 like hashtags, @ mentions, URLs, etc.
        graphs = map(conllu.parse_tree, batches)
        trees = map(_add_root, graphs)
        return trees
