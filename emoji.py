# coding: utf8
# Taken from https://github.com/ines/spacymoji/blob/6bc4e44545d50d642baaba8b5ad36a050e1eb221/spacymoji/__init__.py
from __future__ import unicode_literals

from spacy.tokens import Doc, Span, Token
from spacy.matcher import PhraseMatcher

EMOJIS = [
    "=)",
    ";)",
    ":/",
    ":(",
    ":)",
    "(:",
    "):",
]

class Emoji(object):
    def __init__(self, nlp, emojis=EMOJIS, pattern_id='EMOJI'):
        self.matcher = PhraseMatcher(nlp.vocab)
        emoji_patterns = [nlp(emoji) for emoji in emojis]
        self.matcher.add(pattern_id, None, *emoji_patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []  # keep spans here to merge them later
        for _, start, end in matches:
            spans.append(doc[start : end])
        return doc
