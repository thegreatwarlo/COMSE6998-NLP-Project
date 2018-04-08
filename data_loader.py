# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:22:50 2018

@author: Carlo
"""
import yaml
import praw
from prices import crypto_history

with open('./secrets.yaml', 'r') as secrets_file:
    data = yaml.load(secrets_file)
    
client_id = data['reddit']['api']['client_id']
client_secret = data['reddit']['api']['client_secret']

reddit = praw.Reddit(client_id = client_id,
                     client_secret = client_secret,
                     user_agent='testscript by /u/thegreatwarlo')

def main():    
    print(crypto_history.gather('20170101', '20170102', ['ethereum']))

if __name__=='__main__':
    main()