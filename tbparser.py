import conllu
import os

class TweeboParser(object):
    def __init__(self, tbparser_root, tweets_filename):
        tbparser_root = tbparser_root.rstrip('/')
        tbparser_root = os.path.abspath(tbparser_root)
        tweets_filename = os.path.join(tbparser_root, tweets_filename)
        self._tbparser_root = tbparser_root
        self._tweets_filename = tweets_filename

        # Run TweeboParser install script
        os.system(f'{tbparser_root}/install.sh')

    def parse_tweets(self, tweets):
        with open(self._tweets_filename, 'w') as tweets_file:
            contents = '\n'.join(tweets)
            tweets_file.write(contents)

        # Run CMU's parser
        os.system(f'{self._tbparser_root}/run.sh {self._tweets_filename}')

        # Parse output file, which is formatted in CoNLL-X
        # Since it doesn't use the PHEAD or PDEPREL fields, we can use a CoNLL-U parser library
        output_filename = f'{self._tweets_filename}.predict'
        with open(output_filename, 'r') as output_file:
            contents = output_file.read().strip()
        batches = contents.split('\n\n')
        trees = map(conllu.parse_tree, batches)
        return trees