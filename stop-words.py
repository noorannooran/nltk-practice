from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#stop words: fluff "and, the"

example_sentence = "This is an example showing off stopword filtration."

stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

#oneliner version

filtered_sentence2 = [w for w in words if not w in stop_words]

print(filtered_sentence2)
