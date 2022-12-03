# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:19:57 2022

@author: spika
"""
import numpy as np
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import string
from nltk.stem import *
from nltk.stem.porter import *
from collections import Counter 
#library that contains punctuation
string.punctuation
#nltk.download('stopwords')



#defining the function to remove punctuation
def remove_punctuation(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    inp_str = text
    no_punc = ""
    for char in inp_str:   
        if char not in punctuations:       
            no_punc = no_punc + char
    return no_punc

def apply_tokenizing(text):
    tokenized_text = word_tokenize(text)
    return tokenized_text

def applyStopWords(doc):
    removeStopWords = [word for word in doc if not word in stopwords.words()]
    return removeStopWords

def applyStemmer(doc):
    porter = PorterStemmer()
    singles_porter = [porter.stem(word) for word in doc]
    return singles_porter



#Import CSV (one original and one copy)
file = open('reviews1.csv',errors="ignore"  )
reader = csv.reader(file, delimiter = ',')
data_original = list(reader)


## remove remove_punctuation from copy and insert it in new_data(dictionary)

new_data = []

for i in range(1,len(data_original)):
    review = remove_punctuation(data_original[i][1])
    new_data.append(review)



alldocs_tokenized = []
#REMOVE punctuation from a list of docs
for index, i in enumerate(new_data):
    doc = i
    text = apply_tokenizing(doc)
    alldocs_tokenized.append(text)

#lowe CAse all words
for i in alldocs_tokenized:
    for j in range(0,len(i)):
        i[j] =  i[j].lower()

#Stopwords removal
for x in range(len(alldocs_tokenized)):
    stopWordReview = applyStopWords(alldocs_tokenized[x])
    alldocs_tokenized[x] = stopWordReview

for j in range(len(alldocs_tokenized)):
    alldocs_tokenized[j] = applyStemmer(alldocs_tokenized[j])


#-----------------------------Reverse index TF-IDF-----------------------------
#create a list of all words:
l = alldocs_tokenized

flatten = [ item for sublist in l for item in sublist ]

words = flatten

wordsunique = set(words)
wordsunique = list(wordsunique)


#from textblob import TextBlob as tb

def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum( 1 for doc in doclist if word in doc )

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)) )


def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))


#Create a dictionary of words 
#takes time

tokenized_docs = alldocs_tokenized
#{word: [[doc indexer, [word pos], tf-idf]]}
wordsdc = {}

for doc in tokenized_docs:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = tokenized_docs.index(doc)
            #TODO:
            #positions = list(np.where(np.array(tokenized_docs[index]) == word)[0])
            idfs = tfidf(word, doc, tokenized_docs)
            try:
                wordsdc[word].append([index, idfs])
            except:
                wordsdc[word] = []
                wordsdc[word].append([index, idfs])
#{word: [[doc indexer, [word pos], tf-idf]]}
#worddic['gaming']
#[[0, [23], -3.5123531767879854e-05], [1, [44], -0.00010842481545736825]]

#_--------------------------------------Search engine--------------------------

def search(query):
    try:
        #split sentence into individual words
        query = query.lower()
        try:
            words = query.split(' ')
        except:
            words = list(words)
        #enddic = {}
        idfdic = {}
        closedic = {}
        
        #remove words if not in worddic
        realwords = []
        for word in words:
            if word in list(wordsdc.keys()):
                realwords.append(word)
        
        words = realwords
        numwords = len(words)
        
        
        #make metric of number of occurances of all words in each doc and largest total idf
        for word in words:
            for w in wordsdc[word]:
                index = w[0]
                #amount = len(indpos[1])
                idfscore = w[1]
                #enddic[index] = amount
                idfdic[index] = idfscore
                #fullcount_order = sorted(enddic.items(), key= lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)
                
        '''
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
        '''
        return(query, words, fullidf_order)
        #return(query, words, fullcount_order, combocount_order, fullidf_order)
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

def rank(query):
    search_results = search(query)
    
    #get metrics
    #num_score = search_results[2]
    #per_score = search_results[3]
    tfscore =  search_results[2]

    docs_r = []
    
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
    
    #top five based on tfidf
        for index in range(0,5):
            docs_r.insert(len(docs_r), tfscore[index][0])

        
    
    #single word searched
    except:
        othertops = [tfscore[0][0]]
        
        
        for top in othertops:
            if top not in docs_r:
                docs_r.insert(len(docs_r), top)
                
        
    for index, results in enumerate(docs_r):
        if index < 5:
            print('RESULT', index+1, ":", new_data[results][:])
    
    























