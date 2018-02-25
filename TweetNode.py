class TweetNode(object):
    def __init__(self):
        pass

    def __init__(self, token):
        self._token = token
        for child in token.children:
            self.add_child(child)

    @property
    def children(self):
        return self._children or []

    def add_child(self, child):
        self._children = self._children or []
        child = TweetNode(child)
        self._children.append(child)
