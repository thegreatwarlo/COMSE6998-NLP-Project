# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:22:50 2018

@author: Carlo
"""
import praw

reddit = praw.Reddit(#specify client_id and client_secret
                     user_agent='testscript by /u/thegreatwarlo')

print(reddit.read_only)


for submission in reddit.subreddit('cryptocurrencies').hot(limit=10):
    print(submission.title)