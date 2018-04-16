# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:11:19 2018

@author: Carlo
"""

import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from collections import defaultdict


IN_FILE = './data/bitcoin_markets_daily_discussion_03172018_04082018.csv'
LEXICON = './combined.txt'


    

vader = SIA(lexicon_file = LEXICON)

corpus = []

with open(IN_FILE) as f:
    reader = csv.reader(f, delimiter = ',', quotechar = '"')
    next(f)
    for row in reader:
        ss_score = vader.polarity_scores(row[2])
        out_row = [entry for entry in row]
        out_row.append(ss_score)
        corpus.append(out_row)
        
comments_per_day = defaultdict(list)

for c in corpus:
    comments_per_day[c[1]].append(corpus)
    
features = []

for c in corpus:
    datum = {}
    datum['date'] = c[1]
    datum['sentiment'] = c[4]
    datum['n_comments'] = len(comments_per_day[c[1]])
    datum['length'] = len(c[2])
    features.append(datum)
        


#with open('./secrets.yaml', 'r') as secrets_file:
#    API_keys = yaml.load(secrets_file)

#client_id = API_keys['reddit']['api']['client_id']
#client_secret = API_keys['reddit']['api']['client_secret']

#reddit = praw.Reddit(client_id = client_id,
#                     client_secret = client_secret,
#                     user_agent='testscript by /u/thegreatwarlo')

#print(reddit.read_only)

#bitcoin_markets = reddit.subreddit('BitcoinMarkets')



#    for line in f:
#        if len(line.strip().split('\t')[0:2]) != 2:
#            print line

#data_load = nltk.data.load(LEXICON).split('\n')

#for line in data_load:
#    if len(line.strip().split('\t')[0:2]) != 2:
#        print line