
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



#Import CSV (one original and one copy)
file = open('reviews.csv', errors="ignore")
file1 = open('reviews.csv', errors="ignore")
reader = csv.reader(file, delimiter = ',')
reader1 = csv.reader(file1, delimiter = ',')
data_original = list(reader1)
data = list(reader)


## remove remove_punctuation from copy and insert it in new_data
new_data = data
for i in range(1,len(data)):
    #new_data[data[i][0]] = data[i][1]
    new_data[i][1] = remove_punctuation(new_data[i][1])
        
        
    


