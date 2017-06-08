from nltk.tokenize import sent_tokenize, word_tokenize
# tokenizing = grouping things
# word tokenizers / sentence tokenizer
# separate by word / separate by sentence
# lexicon / corporas
# dictionary (words and meanings) / body of text

example_text = "Hello, Mr.Smith, how are you? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

print(sent_tokenize(example_text))

word_example = word_tokenize(example_text)

for i in word_example:
    print(i)
