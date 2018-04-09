# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:22:50 2018

@author: Carlo
"""
import yaml
import time
import csv
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

from prices import crypto_history

REMOVED = '[removed]'

with open('./secrets.yaml', 'r') as secrets_file:
    data = yaml.load(secrets_file)

client_id = data['reddit']['api']['client_id']
client_secret = data['reddit']['api']['client_secret']

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent='testscript by /u/thegreatwarlo')

bitcoin_markets = reddit.subreddit('BitcoinMarkets')
vader = SIA()

def main():
    # print(crypto_history.gather('20170101', '20170102', ['ethereum']))
    with open('./data/bitcoin_markets_daily_discussion_v1.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['comment_id', 'date', 'body', 'parent_id', 'vader_scores'])
        for submission in get_daily_discussion(300):
            title = submission.title
            date = title.split(']')[-1].strip()
            print('\nSubmission Date: {0}'.format(date))
            non_removed_comment_count = 0
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if comment.body != REMOVED:
                    body = comment.body.replace('\n', '\t')
                    scores = vader.polarity_scores(comment.body)
                    csv_row = [comment.id, date, body, comment.parent().id, scores]
                    writer.writerow(csv_row)
                    non_removed_comment_count += 1
                    time.sleep(1)
            print("Number of comments: {0}".format(non_removed_comment_count))


def get_daily_discussion(limit=100):
    for submission in bitcoin_markets.search('flair:Daily Discussion', sort='new', limit=100):
        yield submission

if __name__=='__main__':
    main()
