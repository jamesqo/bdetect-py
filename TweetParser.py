import spacy

from TweetTree import TweetTree

class TweetParser(object):
    def __init__(self):
        self._nlp = spacy.load('en')

    def tree(self, text):
        return TweetTree(self._nlp(text))
