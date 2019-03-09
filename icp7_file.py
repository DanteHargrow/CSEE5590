from bs4 import BeautifulSoup
import requests
from nltk.util import ngrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag,ne_chunk
import nltk

url = 'https://en.wikipedia.org/wiki/Google'
the_request = requests.get(url)
data = the_request
soup = BeautifulSoup(data.content,"html.parser")


#extracts the following website inputed above
file = open("input.txt","w")
file.write(str(soup.text))
file.close()
file = open("input.txt","r")


# #tokenize
# for i in file:
#     i = file.readline()
#     wtokes = nltk.word_tokenize(i)
#     print(wtokes)
#
# file.seek(0)
#
# #POS
#
# print("POS----------------")
# for p in file:
#     p = file.readline()
#     tokes = nltk.word_tokenize(p)
#     print(nltk.pos_tag(tokes))
#
# file.seek(0)
#
# print("stem--------------------------")
# pStemmer = PorterStemmer()
# for r in file:
#     r = file.readline()
#     print(pStemmer.stem(r))
#
# file.seek(0)
#
# #lemmatizer
# print("lemmatizer-----------------------")
#
# lemmatizer = WordNetLemmatizer()
# for k in file:
#     k = file.readline()
#     print(lemmatizer.lemmatize(k))
#
# file.seek(0)
#
# print("trigrams---------------------------")
# #trigrams
# for j in file:
#     j = file.readline()
#     wordtoke = nltk.word_tokenize(j)
#     trigrams = ngrams(wordtoke,3)
#     for i in trigrams:
#         print(i)
#
# file.seek(0)
#
# print("named entiry recognition----------------------")
#named entity recognition
for h in file:
    h=file.readline()
    print(ne_chunk(pos_tag(wordpunct_tokenize(h))))