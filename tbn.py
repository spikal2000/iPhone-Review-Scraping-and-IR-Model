
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




