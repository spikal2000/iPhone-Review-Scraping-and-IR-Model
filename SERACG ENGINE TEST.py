# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:18:44 2022

@author: spika
"""
import pandas as pd
import numpy as np
import string
import random

import nltk
from nltk.corpus import brown
from nltk.corpus import reuters

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


data = ["""I stumbled on this pc while looking for parts to build a new computer 
The price of this thing is a steal at 1100 dollars 11th gen I7 3060ti watercooled It was a great deal So after it arrived 
Ive been playing on it for over 2 gaming weeks It has been running beautifully it plays Elden Ring on Ultra just fine 
Cyberpunk on high settings at a consistent 60fps and Overwatch 2 on ultra ring I play on 1080p not 4k keep that in mind 
My cpu temps run just fine as well it averages rings around 50c with a workload on it and gpu temps are around 60c Max Ive seen them run is 70c 
but I went into MSI Center and ramped my fans up to 100 and they ultra lowered quite a bit Now the downfall of this pc is the RAM I was promised 
16gb of 3200mhz ram It currently sits at 2667mhz as I am typing this My friend has this exact PC and his runs at 3200 So Im not sure whats going 
on here but my RAM is definitely off I have ordered 32gb of 3200mhz that is on the way Which I planned on upgrading the ram anyway but i had to buy 
4 sticks instead of 2 sticks Also I do not trust this watercooler or Power supply Both are off brand and I plan on replacing them 
before they go out I know I have said some pretty big cons in this review however I think this is a great pc so far It hasnt let me down 
yet the m2 is fast as hell I elder ring ultra really like it but I do plan on making upgrades to it soon for longevity and efficiency""", 
       """I got this at a huge discount 3060ti watercooled  1200 but it wouldnt be worth it at full price AIO was scuffed up which had me worried
       CPU was cpu damaged and wouldnt 3060ti watercooled 3060ti watercooled  even boot but it did one of the fans on AIO radiator has a decent wobble to it One of the display port
       connections in the graphics card  elder ring ultra is extremely temperamental and goes off for like no reason at allAside from that its starts runs and 
       operates like anything you could ever expect out of a base model gaming computer"""]
       
       
#{!, ", #}
exclude = set(string.punctuation)
alldocslist = []


#REMOVE punctuation from a list of docs
for index, i in enumerate(data):
    text = i
    text = ''.join(ex for ex in text if ex not in exclude)
    alldocslist.append(text)
    #print(index,"AAAAAAAAAAAAAAAAAAAAAAAAAAAA", i)

plot_data = [[]] * len(alldocslist)
for doc in alldocslist:
    text = doc
    tokentext = word_tokenize(text)
    plot_data[index].append(tokentext)

#first index gives all documents, 2nd index gives specific document, 3rd index gives the words of that document
plot_data[0][1][0:10]

#make all words lower case
for x in range(len(plot_data)):
    lowers = [ word.lower() for word in plot_data[0][x] ]
    plot_data[0][x] = lowers

#remove stopwords from all docs
stop_words = set(stopwords.words('english'))
stem_words = [[]] * len(plot_data)
for x in range(len(plot_data)):
    filtered_sentence = [ w for w in plot_data[0][x] if not w in stop_words ]
    #stem words 
    porter_stemmer = PorterStemmer()
    stemmed_sentence = [ porter_stemmer.stem(w) for w in filtered_sentence ]
    stem_words[index].append(stemmed_sentence)
    plot_data[0][x] = filtered_sentence

#-----------------------------Reverse index TF-IDF-----------------------------
#create a list of all words:
l = plot_data[0]

flatten = [ item for sublist in l for item in sublist ]

words = flatten

wordsunique = set(words)
wordsunique = list(wordsunique)

import math 
#from textblob import TextBlob as tb

def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum( 1 for doc in doclist if word in doc )

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)) )


def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))


import re 
import numpy as np

#Create a dictionary of words 
#takes time

plottest = plot_data[0][:]

worddic = {}

for doc in plottest:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = plottest.index(doc)
            positions = list(np.where(np.array(plottest[index]) == word)[0])
            idfs = tfidf(word, doc, plottest)
            try:
                worddic[word].append([index, positions, idfs])
            except:
                worddic[word] = []
                worddic[word].append([index, positions, idfs])
#{word: [[doc indexer, [word pos], tf-idf]]}
#worddic['gaming']
#[[0, [23], -3.5123531767879854e-05], [1, [44], -0.00010842481545736825]]


#_--------------------------------------Search engine--------------------------

from collections import Counter 

def search(query):
    try:
        #split sentence into individual words
        query = query.lower()
        try:
            words = query.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
        closedic = {}
        
        #remove words if not in worddic
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)
        
        words = realwords
        numwords = len(words)
        
        
        #make metric of number of occurances of all words in each doc and largest total idf
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key= lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)
                
        
        #make metric of what percentage of words appear in each doc
        combo = []
        allocations = {k: worddic.get(k, None) for k in (words)}
        for worddex in list (allocations.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], reverse=True)
        
        return(query, words, fullcount_order, combocount_order, fullidf_order)
        '''
        0 query-> search query
        1 words -> actual words searched
        2 fullcount_order -> num occur
        3 combocount_order - > % of terms 
        4 fullidf_order - > tf-idf
        
        '''
    except:
        return("")



#-----------------------Rank and return---------------

def rank(term):
    results = search(term)
    
    #get metrics
    num_score = results[2]
    per_score = results[3]
    tfscore =  results[4]

    final_candidates = []
    
    #rule1: if high word order score & 100% percentage terms then put at top position
    try: 
        '''
        first_candidates = []
        
        for candidates in order_score:
            if candidates[1] > 1:
                first_candidates.append(candidates[0])
                
        second_candidates = []
        
        for match_candidates in per_score:
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])
        '''
    #rule3: add top tf-idf results
        final_candidates.insert(len(final_candidates), tfscore[0][0])
        final_candidates.insert(len(final_candidates), tfscore[1][0])
    
    #single word searched
    except:
        othertops = [tfscore[0][0]]
        
        
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates), top)
                
        
    for index, results in enumerate(final_candidates):
        if index < 5:
            print('RESULT', index+1, ":", alldocslist[results][:])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










