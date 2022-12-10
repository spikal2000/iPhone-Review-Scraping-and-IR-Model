# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:18:29 2022

@author: spika
"""

import csv
import requests
from bs4 import BeautifulSoup


def getAmazonSearch(query):
    amazon_url = "https://www.amazon.com/s?k="+ query
    page = requests.get(amazon_url, headers = header)
    return page


#Get context of individual product pages with data-asin 
def searchAsin(asin):
    url = "https://www.amazon.com/dp/" + asin
    page = requests.get(url, headers = header)
    return page

#Extract the product names
def getProductNames(asin):
    url = "https://www.amazon.com/dp/" + asin
    page = requests.get(url, headers = header)
    soup = BeautifulSoup(page.content, "lxml")
    product_name = " "
    for i in soup.findAll("span", {'class': 'a-size-large product-title-word-break'}):
        product_name = i.text
    return product_name

#to see all Reviews link and extract content
def searchReviews(reviewLink):
    url = "https://www.amazon.com" + reviewLink
    page = requests.get(url, headers = header)
    return page


#query = "graphics+cards"
query = "iphone"
#url = amazon_url + query

#Without the header amazon will not give me access
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36 ',
          'referer': 'https://www.amazon.com/s?k=iphone&crid=2863RYB3M0F16&sprefix=iphone+14%2Caps%2C257&ref=nb_sb_noss_2'}

#https://www.amazon.com/s?k=iphone&crid=399XGN9Z3Y6UG&sprefix=ipho%2Caps%2C336&ref=nb_sb_noss_2

##GPU https://www.amazon.com/s?k=graphics+cards&ref=nb_sb_noss


#&page=4&qid=1666777740&ref=sr_pg_4
#extract the asin numbers 
asins = []
for j in range(1):
    page = getAmazonSearch(query + '&page='+ str(j) + '&qid=1666777740&ref=sr_pg_'+ str(j))
    soup = BeautifulSoup(page.content, "lxml") 
    for i in soup.findAll("div", {'class': 's-result-item s-asin sg-col-0-of-12 sg-col-16-of-20 sg-col s-widget-spacing-small sg-col-12-of-16'}):
        asins.append(i['data-asin'])
    

titles = []
for i in range(0,len(asins)):
    page = searchAsin(asins[i])
    soup = BeautifulSoup(page.content, "lxml")
    for j in soup.findAll("span", {'id': "productTitle"}):
        titles.append(j.text)
    
    
links = {}
for i in range(0,len(asins)):
    page = searchAsin(asins[i])
    soup = BeautifulSoup(page.content, "lxml")
    for j in soup.findAll("a", {'data-hook': "see-all-reviews-link-foot"}):
        links[titles[i]] = j['href']

        

reviews = {}
for key, value in links.items():
    for j in range(1):
        page = searchReviews(value + '&pageNumber=' + str(j))
        soup = BeautifulSoup(page.content, "lxml")
        for k in soup.findAll("span", {'data-hook' : 'review-body'}):
            if key not in reviews.keys():
                reviews[key] = [k.text]
            else:
                reviews[key].append(k.text)

#call the getProductNames and store the names in a list
products = {}
for p in reviews.keys():
    #products.append(getProductNames(i))
    products[p] = getProductNames(p)

temp_reviews = []
for i in reviews:
    for j in range(0, len(reviews[i])):
        temp_reviews.append({'title': i, 'review':reviews[i][j]})

#extract csv file
fieldNames = ['title', 'review']
with open('reviews1.csv', 'w', encoding='UTF8', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldNames)
    writer.writeheader()
    writer.writerows(temp_reviews)
