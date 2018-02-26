import spacy

class TweetParser(object):
    def __init__(self):
        self._nlp = spacy.load('en')

    def tree(self, text):
        return TweetTree(self._nlp(text))

class TweetTree(object):
    def __init__(self, doc):
        self._doc = doc
        self.root = self._create_root()

    def _create_root(self):
        root = TweetNode()
        for sent in self._doc.sents:
            root.add_child(sent.root)
        return root

class TweetNode(object):
    def __init__(self, token=None):
        self._children = None
        self._token = token
        if token is not None:
            for child in token.children:
                self.add_child(child)

    @property
    def children(self):
        return self._children or []

    def add_child(self, child):
        self._children = self._children or []
        child = TweetNode(child)
        self._children.append(child)
