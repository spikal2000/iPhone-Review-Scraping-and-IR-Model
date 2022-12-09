# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:24:20 2022

@author: spika
"""

#pip install sentence-transformers
#nvidia-smi 
#pip install faiss-cpu
#pip install bert-score


import pandas as pd
import time
from tqdm import tqdm
import seaborn as sns
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')
import gc


data = ["""I stumbled on this pc while Elder ring can play looking for parts to build a new computer. The price of this thing is a steal at 1100 dollars. 11th gen I7, 3060ti, water-cooled. It was a great deal. So, after it arrived, I've been playing on it for over 2 weeks. It has been running beautifully, it plays Elden Ring on Ultra just fine, Cyberpunk on high settings at a consistent 60fps, and Overwatch 2 on ultra (I play on 1080p not 4k, keep that in mind). 
        My cpu temps run just fine as well, it averages around 50c with a workload on it and gpu temps are around 60c. Max I've seen them run is 70c but I went into MSI Center and ramped my fans up to 100% and they lowered quite a bit. Now, the downfall of this pc is the RAM. I was promised 16gb of 3200mhz ram. It currently sits at 2667mhz as I am typing this. My friend has this exact PC and his runs at 3200. So I'm not sure what's going on here but my RAM is definitely off. I have ordered 32gb of 3200mhz that is on the way. Which I planned on upgrading the ram anyway, but i had to buy 4 sticks instead of 2 sticks. Also, I do not trust this water-cooler or Power supply. Both are off brand and I plan on replacing them before they go out. I know I have said some pretty big cons in this review, however, I think this is a great pc so far. It hasn't let me down yet, the m.2 is fast as hell. I really like it, but I do plan on making upgrades to it soon for longevity and efficiency.""",
"""UPDATE: Support did get back to me and in the process of getting information about the RAM modules, they are not both working and I have the advertised 16gb. I've updated the rating to four stars because support was still quite slow to get back to me and there wasn't a quicker, more direct way to contact them. The computer works as intended now and the damaged case isn't the manufacturer's fault.ORIGINAL REVIEW:Arrived with some cosmetic damage to the top front of the case. This isn't the seller/manufacturer's fault but certainly worth noting. It runs fine and has a lot of performance for the money. The reason it gets one star is because Windows is only reporting 8gb of RAM, and it's supposed to have 16. I checked to make sure both sticks are seated but still only showing 8gb. I contacted support through the only method possible - a form on their website. I've heard nothing back besides the automated reply. It's been five days. I am also unable to register it because it seems like the model is discontinued and the model isn't an option on the registration form. Overall the computer is fine, notwithstanding defective RAM. However, five days and no reply after I've received a defective product is unacceptable and I cannot recommend this  computer based on that alone."""]
df = pd.DataFrame({'review':data})


# df.dropna(inplace=True)
# import faiss
# encoded_data = model.encode(df.review.tolist())
# encoded_data = np.asarray(encoded_data.astype('float32'))
# index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
# index.add_with_ids(encoded_data, np.array(range(0, len(df))))
# faiss.write_index(index, 'review.index')

# from pprint import pprint



# def fetch_movie_info(dataframe_idx):
#     info = df.iloc[dataframe_idx]
#     meta_dict = {}
#     # meta_dict['Title'] = info['Title']
#     # meta_dict['Plot'] = info['Plot'][:500]
#     meta_dict['review'] = info['review']
#     return meta_dict

# def search(query, top_k, index, model):
#     t=time.time()
#     query_vector = model.encode([query])
#     top_k = index.search(query_vector, top_k)
#     print('>>>> Results in Total Time: {}'.format(time.time()-t))
#     top_k_ids = top_k[1].tolist()[0]
#     top_k_ids = list(np.unique(top_k_ids))
#     results =  [fetch_movie_info(idx) for idx in top_k_ids]
#     return results

# query="Ram modules are good?"
# results=search(query, top_k=5, index=index, model=model)

# # print("\n")
# # for result in results:
# #     print('\t',pprint(result))

# ## Load our cross-encoder. Use fast tokenizer to speed up the tokenization
# from sentence_transformers import CrossEncoder
# cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)


# def cross_score(model_inputs):
#     scores = cross_model.predict(model_inputs)
#     return scores


# model_inputs = [[query,item['review']] for item in results]

# scores = cross_score(model_inputs)
# #Sort the scores in decreasing order
# ranked_results = [{'review': inp['review'], 'Score': score} for inp, score in zip(results, scores)]
# ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)


# # print("\n")
# # for result in ranked_results:
# #     print('\t',pprint(result))

import faiss

encoded_data = model.encode(df.review.tolist())
encoded_data = np.asarray(encoded_data.astype('float32'))
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.array(range(0, len(df))))
faiss.write_index(index, 'review.index')


def fetch_movie_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    meta_dict = {}
    # meta_dict['Title'] = info['Title']
    # meta_dict['Plot'] = info['Plot'][:500]
    meta_dict['review'] = info['review']
    return meta_dict

def search(query, top_k, index, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_movie_info(idx) for idx in top_k_ids]
    return results

query="Can you play elder ring with 3060"
results=search(query, top_k=5, index=index, model=model)

import bert_score
bert_score.__version__
from bert_score import score

ref=["Can you play elder ring with 3060"]

ranked_results_bert = []

for cand in results:
    P, R, F1 = score([cand['review']], ref, lang='en')
    ranked_results_bert.append({'review': cand['review'], 'Score': F1.numpy()[0]})

#Sort the scores in decreasing order
ranked_results_bert = sorted(ranked_results_bert, key=lambda x: x['review'], reverse=False)
print("\n")
for result in ranked_results_bert:
    print('\t',result)


final_results = pd.DataFrame()
#final_results['faiss_ranking'] = [item['review'] for item in results]
final_results['bert_score'] = [item['review'] for item in ranked_results_bert]
     





