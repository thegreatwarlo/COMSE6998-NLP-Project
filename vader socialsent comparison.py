# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:26:05 2018

@author: Carlo
"""
import csv

vader_txt = './vader_lexicon.txt'
socialsent_txt = './socialsent_Bitcoin.txt'



with open(vader_txt) as f:
    vader = set([line.strip().split('\t')[0] for line in f])
    
    
with open(socialsent_txt) as f:
    combined = [line.strip().split('\t')[0:2] for line in f]

conflict = set()

for t in combined:
    if t[0] in vader:
        conflict.add(t[0])
    
with open(vader_txt) as f:
    for line in f:
        token_val = line.strip().split('\t')[0:2]
        if token_val[0] not in conflict:
            combined.append(token_val)
            
            
#just as a test
check = set([token[0] for token in combined])

print len(check)


with open('combined.txt' , 'wb') as f:
    writer = csv.writer(f, delimiter = '\t')
    for row in combined:
        writer.writerow(row)