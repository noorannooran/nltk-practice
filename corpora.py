from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

print(tok[5:15])

sample2 = gutenberg.raw("blake-poems.txt")
tok2 = sent_tokenize(sample2)

print(tok2[5:10])
