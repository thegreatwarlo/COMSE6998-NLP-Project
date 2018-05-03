# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:11:19 2018

@author: Carlo
"""
#%%
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from datetime import datetime
import pandas as pd
import numpy as np
#from collections import defaultdict

#%%
LEXICON = './combined.txt'

socialsent = SIA(lexicon_file = LEXICON)
vader = SIA()


def read_corpus(infile, corp, mode):
    with open(infile) as f:
        reader = csv.reader(f, delimiter = ',', quotechar = '"')
        #skip first row
        next(f)
        for row in reader:
            #check the row has the expected number of fields
            if len(row) == 9:
                #compute the socialsent scores from the body
                ss_score = socialsent.polarity_scores(row[3])
                #copy the row in output
                out_row = [entry for entry in row]
                #replace non-bitcoin user flairs
                if mode != 'bitcoin':
                    out_row[-1] = 'None'
                out_row.append(ss_score)
                corp.append(out_row)

def flatten(lexicon, d):
    d = pd.concat([d.drop(['%s_scores' % (lexicon)], axis=1), 
                       d['%s_scores' % (lexicon)].apply(pd.Series)], axis=1)
    d = d.rename(index = str, columns = 
                 {"pos": "%s_pos" % (lexicon), "neg": "%s_neg" % (lexicon), 
                  "neu": "%s_neu" % (lexicon), "compound":"%s_compound" % (lexicon)})
    
    
def sentiment_normal_avg_byday(df, lexicon_name):
    # lexicon name can only be "vader", "socialsent".
    sent_catogory = ["compound", "neu", "neg", "pos"]
    col_names = [lexicon_name+"_"+i for i in sent_catogory]
    groupby_dict = {}
    for key in col_names:
        groupby_dict[key] = "mean"
    return df.groupby("daily_discussion_date").agg(groupby_dict)

def sentiment_votes_avg_byday(df, lexicon_name):
    weighted_mean = lambda x: np.average(x, weights=df.loc[x.index, "upvotes"])
    # lexicon name can only be "vader", "socialsent".
    sent_catogory = ["compound", "neu", "neg", "pos"]
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
    sent_catogory = ["compound", "neu", "neg", "pos"]
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
    for t in opinion_types:
        dataframe[t] = 0
    for index, entry in dataframe.iterrows():
        dataframe.set_value(index, entry["author_opinion"], 1)
    return dataframe

def opinion_avg_byday(df):
    opinion_types = ['Bullish','Bearish','Long-term Holder','Bitcoin Skeptic','None']
    groupby_dict = {}
    for t in opinion_types:
        groupby_dict[t] = "mean"
    return df.groupby("daily_discussion_date").agg(groupby_dict)



#%%
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_august_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_september_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_october_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_november_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_december_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_january_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_february_v1.csv'
#IN_FILE = './data/data/bitcoin_markets_daily_discussion_march_v1.csv'
IN_FILE = './eth/eth/ethtrader_daily_discussion_september_v1.csv'

#this are use to distinguish between the different type of input files
out_tags= IN_FILE.split('/')[-1].split('_')

out_file = '%s_features_%s.csv' % (out_tags[0], out_tags[-2]) 

#%%
# main function
corpus = []  

read_corpus(IN_FILE, corpus, out_file[0])

#%%
data = pd.DataFrame(corpus)
data.columns = ["comment_id","daily_discussion_date","created","body",
                     "parent_id","vader_scores","upvotes","downvotes",
                     "author_opinion","socialsent_scores"]

#flattens the socialsent scores
data = pd.concat([data.drop(['socialsent_scores'], axis=1), 
                       data['socialsent_scores'].apply(pd.Series)], axis=1)

#renames the socialsent columns
data = data.rename(index = str, columns = 
                 {"pos": "socialsent_pos", "neg": "socialsent_neg", 
                  "neu": "socialsent_neu", "compound":"socialsent_compound"})

for index, row in data.iterrows():
    #convert vader score to entries
    if 'vader_scores' in row:
        data.set_value(index, 'vader_scores', eval(row['vader_scores']))
    
    #parse date
    dt = datetime.fromtimestamp(int(row['created'].split('.')[0])).strftime('%Y-%m-%d %H')
    data.set_value(index, 'daily_discussion_date', dt)
 
#flattens the vader scores
data = pd.concat([data.drop(['vader_scores'], axis=1), 
                       data['vader_scores'].apply(pd.Series)], axis=1)

#renames vader columns
data = data.rename(index = str, columns = 
                 {"pos": "vader_pos", "neg": "vader_neg", 
                  "neu": "vader_neu", "compound":"vader_compound"})
   
# get the number of next-layer comments
data["num_child"] = data["comment_id"].apply(
        lambda x: data[data["parent_id"]==x].shape[0])

# sentiment features:
print("extracting sentiment features....")
daily_feature = sentiment_normal_avg_byday(data, "vader")
daily_feature = pd.concat([daily_feature, sentiment_normal_avg_byday(data, "socialsent")], axis=1)
#daily_feature = pd.concat([daily_feature, sentiment_votes_avg_byday(data, "vader")], axis=1)
#daily_feature = pd.concat([daily_feature, sentiment_votes_avg_byday(data, "socialsent")], axis=1)
# get the number of next-layer comments
#daily_feature = pd.concat([daily_feature, sentiment_child_avg_byday(data, "vader")], axis=1)


# statistical features:
print("extracting statistical features....")
data = get_author_opinion(data)

df_num_comments = data.groupby("daily_discussion_date").size().to_frame()
df_num_comments.columns = ["num_comments"]

df_num_children = data.groupby("daily_discussion_date").agg({"num_child":"mean"})

data['num_word'] = [len(str(x["body"])) for i,x in data.iterrows()]
df_num_word = data.groupby("daily_discussion_date").agg({"num_word":"mean"})

daily_feature = pd.concat([daily_feature, opinion_avg_byday(data), 
                           df_num_comments, df_num_children, df_num_word], axis=1)
    

daily_feature.to_csv('features_feb.csv')





       
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

