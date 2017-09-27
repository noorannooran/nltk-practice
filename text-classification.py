import nltk
import random
from nltk.corpus import movie_reviews

"""
creating own algorithm / "text classifier"
for analysis
can classify for pos/neg sentiment
type of writing eg. political
"""

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print(documents[1])

"""
compile these words
most positive (pos or negative)
and then search other texts for these words
"""

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#print(all_words.most_common(15)) #includes punctuation and filler words

print(all_words["stupid"]) #253 movie reviews include the word stupid


    
