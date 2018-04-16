# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:11:19 2018

@author: Carlo
"""

import yaml
import time
import csv
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


REMOVED = '[removed]'
LEXICON = './combined.txt'

with open('./secrets.yaml', 'r') as secrets_file:
    API_keys = yaml.load(secrets_file)

client_id = API_keys['reddit']['api']['client_id']
client_secret = API_keys['reddit']['api']['client_secret']

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent='testscript by /u/thegreatwarlo')

print(reddit.read_only)

bitcoin_markets = reddit.subreddit('BitcoinMarkets')

for submission in bitcoin_markets.hot(limit=10):
    print(submission.title)
    

socialsent = SIA(lexicon_file = LEXICON)

with open(LEXICON) as f:
    for line in f:
        if len(line.strip().split('\t')[0:2]) != 2:
            print line

data_load = nltk.data.load(LEXICON).split('\n')

for line in data_load:
    if len(line.strip().split('\t')[0:2]) != 2:
        print line