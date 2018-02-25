class TweetTree(object):
    def __init__(self, doc):
        self._doc = doc
        self.root = self._create_root()

    def _create_root():
        root = TweetNode()
        for sent in self._doc.sents:
            root.add_child(sent.root)
        return root
