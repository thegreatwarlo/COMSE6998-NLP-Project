# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:11:19 2018

@author: Carlo
"""

import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from datetime import datetime
import pandas as pd
import numpy as np
#from collections import defaultdict


IN_FILE = './data/bitcoin_markets_daily_discussion_03172018_04082018.csv'
LEXICON = './combined.txt'


    

socialsent = SIA(lexicon_file = LEXICON)
vader = SIA()

corpus = []

with open(IN_FILE) as f:
    reader = csv.reader(f, delimiter = ',', quotechar = '"')
    next(f)
    for row in reader:
        ss_score = socialsent.polarity_scores(row[2])
        out_row = [entry for entry in row]
        out_row.append(ss_score)
        corpus.append(out_row)


    
def sentiment_normal_avg_byday(df, lexicon_name):
    # lexicon name can only be "vader", "socialsent".
    sent_catogory = ["compound", "neutral", "negative", "positive"]
    col_names = [lexicon_name+"_"+i for i in sent_catogory]
    groupby_dict = {}
    for key in col_names:
        groupby_dict[key] = "mean"
    return df.groupby("daily_discussion_date").agg(groupby_dict)

def sentiment_votes_avg_byday(df, lexicon_name):
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, "upvotes"])
    # lexicon name can only be "vader", "socialsent".
    sent_catogory = ["compound", "neutral", "negative", "positive"]
    col_names = [lexicon_name+"_"+i for i in sent_catogory]
    renames = [lexicon_name+"_"+"votes"+"_"+i for i in sent_catogory]
    groupby_dict = {}
    for key in col_names:
        groupby_dict[key] = weighted_mean
    df_result = df.groupby("daily_discussion_date").agg(groupby_dict)
    df_result.columns = renames
    return df_result

def sentiment_child_avg_byday(df, lexicon_name):
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, "num_child"])
    # lexicon name can only be "vader", "socialsent".
    sent_catogory = ["compound", "neutral", "negative", "positive"]
    col_names = [lexicon_name+"_"+i for i in sent_catogory]
    renames = [lexicon_name+"_"+"child"+"_"+i for i in sent_catogory]
    groupby_dict = {}
    for key in col_names:
        groupby_dict[key] = weighted_mean
    df_result = df.groupby("daily_discussion_date").agg(groupby_dict)
    df_result.columns = renames
    return df_result

# Bullish, Bearish, Long-term Holder, Bitcoin Skeptic, None
def get_author_opinion(dataframe):
    opinion_types = ['Bullish','Bearish','Long-term Holder','Bitcoin Skeptic','None']
    for type in opinion_types:
        dataframe[type] = 0
    for i in range(dataframe.shape[0]):
        dataframe.set_value(i, dataframe.iloc[i]["author_opinion"], 1)
    return dataframe

def opinion_avg_byday(df):
    opinion_types = ['Bullish','Bearish','Long-term Holder','Bitcoin Skeptic','None']
    groupby_dict = {}
    for type in opinion_types:
        groupby_dict[type] = "mean"
    return df.groupby("daily_discussion_date").agg(groupby_dict)

# main function

data_test = pd.DataFrame(corpus)
data_test.columns = ["comment_id","date","body","parent_id","vader_scores","socialsent_scores"]

for index, row in data_test.iterrows():
    #convert vader score to dict
    row['vader_scores'] = eval(row['vader_scores'])
    #parse date
    Month = row['date'].split(',')[1].split(' ')[1][0:3]
    Day = row['date'].split(',')[1].split(' ')[2]
    year = row['date'].split(',')[2].split(' ')[-1]
    row['date'] = datetime.strptime(Month + ' ' + Day + ' ' + year, '%b %d %Y')
    
# get the number of next-layer comments
data_test["num_child"] = data_test["comment_id"].apply(lambda x: data_test[data_test["parent_id"]==x].shape[0])




       
#comments_per_day = defaultdict(list)

#for c in corpus:
#    comments_per_day[c[1]].append(c)
    
#features = []

#for c in corpus:
#    datum = {}
#    datum['date'] = c[1]
#    datum['sentiment'] = c[4]
#    datum['n_comments'] = len(comments_per_day[c[1]])
#    datum['length'] = len(c[2])
#    features.append(datum)
        
#with open('./data/bitcoin_markets_daily_discussion_v2.csv', 'w') as csvfile:
#        writer = csv.writer(csvfile)
#        writer.writerow(['comment_id', 'date', 'body', 'parent_id', 'vader_scores', 'socialsent_scores'])
#        for submission in corpus:
#            writer.writerow(submission)

