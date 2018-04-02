#!/usr/bin/env python

""" 
Sample code of getting tweet JSON objects by tweet ID lists.

You have to install tweepy (This script was tested with Python 2.6 and Tweepy 3.3.0)
https://github.com/tweepy/tweepy
and set its directory to your PYTHONPATH. 

You have to obtain an access tokens from dev.twitter.com with your Twitter account.
For more information, please follow:
https://dev.twitter.com/oauth/overview/application-owner-access-tokens

Once you get the tokens, please fill the tokens in the squotation marks in the
following "Access Information" part. For example, if your consumer key is 
LOVNhsAfB1zfPYnABCDE, you need to put it to Line 33
consumer_key = 'LOVNhsAfB1zfPYnABCDE' 



"""

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# call user.lookup api to query a list of user ids.
import tweepy
import sys
import json
import codecs
import pandas as pd
import os
from tweepy.parsers import JSONParser

####### Access Information #################

# Parameter you need to specify
with open('twitter_creds.txt') as file:
	lines = file.read().strip().splitlines()
consumer_key, consumer_secret, access_key, access_secret = lines

inputFile = 'tweet_id'
outputFile = 'tweet.json'

#############################################
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth_handler=auth, parser=JSONParser())

indf = pd.read_csv('data.csv', encoding='utf-8', names=['tweetid', 'userid', 'class', 'type', 'form', 'teasing', 'author_role', 'emotion'])
indf.insert(2, 'tweettext', pd.Series('', dtype=object))
indf.set_index('tweetid', inplace=True)

with codecs.open(outputFile, 'w', encoding='utf8') as outFile:
	outFile.write('[')
	l = list(indf.index)
	for ln in chunkIt(l, 75):
		rst = api.statuses_lookup(id_=ln)
		for tweet in rst:
			i, txt = tweet['id'], tweet['text']
			indf.loc[i, 'tweettext'] = txt

indf.to_csv('data_withtext.csv', encoding='utf-8')
