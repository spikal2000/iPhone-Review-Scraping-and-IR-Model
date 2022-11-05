import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import string
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
    


#Import CSV (one original and one copy)
file = open('reviews.csv', errors="ignore")
reader = csv.reader(file, delimiter = ',')
data_original = list(reader)


## remove remove_punctuation from copy and insert it in new_data(dictionary)
new_data = {}
for i in range(1,len(data_original)):
    asic = data_original[i][0]
    review = remove_punctuation(data_original[i][1])#calling the remove punctuation function
    if asic in new_data:
        new_data[asic].append(review)
    else:
        new_data[asic] = [review]
        
#check the results 
'''       
for i in new_data.keys():
    for j in range(0,len(new_data[i])):
        print(i, new_data[i][j])
'''  

tokenized_data = {}
#tokenize the new_data results
for i in new_data:
    for j in range(0, len(new_data[i])):
        asic = i
        tokenized_review = apply_tokenizing(new_data[i][j])
        if asic in tokenized_data:
            tokenized_data[asic].append(tokenized_review)
        else:
            tokenized_data[asic] = [tokenized_review]
'''
for i in range(len(s)):    
    s[i] = s[i].lower()
'''
for key in tokenized_data:
    asic = key
    for i in tokenized_data[key]:
        for j in range(0,len(i)):
            i[j] =  i[j].lower()
            


tokenized_stopword_data = {}
for key in tokenized_data:
    asic = key
    for j in tokenized_data[key]:
        stopWordReview = applyStopWords(j)
        if asic in tokenized_stopword_data:
            tokenized_stopword_data[asic].append(stopWordReview)
        else:
            tokenized_stopword_data[asic] = [stopWordReview]
  






