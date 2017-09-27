from nltk.stem import WordNetLemmatizer

#similar to stemming but using synonyms

lemmatizer = WordNetLemmatizer()

##print(lemmatizer.lemmatize("cats"))
##print(lemmatizer.lemmatize("cacti"))
##print(lemmatizer.lemmatize("geese"))
##print(lemmatizer.lemmatize("rocks"))
##print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("better", pos="a")) #default pos is n (noun)

print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", pos="v"))

