
import csv
import string
#library that contains punctuation
string.punctuation


#defining the function to remove punctuation
def remove_punctuation(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    inp_str = text
    no_punc = ""
    for char in inp_str:   
        if char not in punctuations:       
            no_punc = no_punc + char
    return no_punc
#storing the puntuation free text
#data['clean_msg']= data['v2'].apply(lambda x:remove_punctuation(x))
#data.head()



#Import CSV 
file = open('reviews.csv', errors="ignore")
reader = csv.reader(file, delimiter = ',')
data = list(reader)
#data[2][1]
#Out[109]: 'B08W8DGK3X'
#data[1][2]
#Out[112]: "\nUPDATE: Support 

#data = reviews_df.to_dict(orient='dict', into=<class 'dict'>)
#print(df.loc[[159220]])

data_copy = data

#for key, value in data_copy.items():
#    data_copy[]
        



#for key in reviews_df['asin']:
##    for value in reviews_df['review']:
#        data[key] = value
        
    


