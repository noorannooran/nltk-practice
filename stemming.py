from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#stemming : getting the root stem of words

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))

example_sentence = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

words = word_tokenize(example_sentence)

for w in words:
    print(ps.stem(w))
