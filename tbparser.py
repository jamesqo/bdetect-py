import conllu
import nltk
import os

from collections import namedtuple, OrderedDict
from itertools import islice
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from util import exec_and_check, log_call

def _add_root(graph, tweet):
    # Turns a multi-rooted graph into a tree by adding a root node.
    return TreeRoot(children=graph, tweet=tweet)

def _remove_newlines(tweet):
    return tweet.replace('\n', ' ').replace('\r', ' ')

def _scrub_trivia(trees):
    log_call()
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

def _stem(trees):
    log_call()

    def _wordnet_pos(pos):
        if pos == 'A':
            return wordnet.ADJ
        elif pos == 'V':
            return wordnet.VERB
        elif pos == 'N':
            return wordnet.NOUN
        elif pos == 'R':
            return wordnet.ADV

        return wordnet.NOUN

    def do_stem(node):
        assert node.data['lemma'] == '_'

        form, pos = node.data['form'], node.data['upostag']
        lemma = lem.lemmatize(form, pos=_wordnet_pos(pos))
        lemma = lemma.lower()
        node.data['lemma'] = lemma

        for child in node.children:
            do_stem(child)

    if not nltk.download('wordnet', quiet=True):
        raise RuntimeError("Failed to download WordNet corpus")

    lem = WordNetLemmatizer()
    for tree in trees:
        for child in tree.children:
            do_stem(child)

    return trees

class TreeRoot(object):
    def __init__(self, children, tweet):
        self.children = children
        self._tweet = tweet

    def __str__(self):
        return self._tweet

class TweeboParser(object):
    def __init__(self,
                 tbparser_root,
                 input_fname,
                 refresh_predictions=False,
                 scrub_trivia=True,
                 stem=True):
        log_call()
        tbparser_root = tbparser_root.rstrip('/')
        tbparser_root = os.path.abspath(tbparser_root)
        input_fname = os.path.join(tbparser_root, input_fname)

        self._tbparser_root = tbparser_root
        self._input_fname = input_fname
        self._output_fname = '{}.predict'.format(input_fname)
        self._run_scripts = refresh_predictions or not os.path.isfile(self._output_fname)
        self._scrub_trivia = scrub_trivia
        self._stem = stem

        if self._run_scripts:
            # Run TweeboParser install script
            exec_and_check('cd {} && bash install.sh'.format(tbparser_root))

    def parse_tweets(self, tweets):
        log_call()
        if self._run_scripts:
            with open(self._input_fname, 'w', encoding='utf-8') as tweets_file:
                # Twitter permits newlines in tweets, causing problems with the dependency parser
                # which expects one tweet per line in the input file.
                tweets = [_remove_newlines(tweet) for tweet in tweets]
                contents = '\n'.join(tweets)
                tweets_file.write(contents)

            # Run CMU's parser
            # TODO: It hangs for a long time after making the predictions because it generates 4000+
            # files (one per tweet) in working_dir/test_score. Find out how to make it stop that.
            exec_and_check('cd {} && bash run.sh {}'.format(self._tbparser_root, self._input_fname))

        # Parse output file, which is formatted in CoNLL-X
        # Since it doesn't use the PHEAD or PDEPREL fields, we can use a CoNLL-U parser library
        with open(self._output_fname, 'r', encoding='utf-8') as output_file:
            contents = output_file.read().strip()
        batches = contents.split('\n\n')
        batches = islice(batches, len(tweets))
        graphs = map(conllu.parse_tree, batches)
        trees = [_add_root(graph, tweet) for graph, tweet in zip(graphs, tweets)]
        if self._scrub_trivia:
            trees = _scrub_trivia(trees)
        if self._stem:
            trees = _stem(trees)
        return list(trees)
