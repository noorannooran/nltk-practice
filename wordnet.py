from nltk.corpus import wordnet

"""
wordnet is the largest capability with nltk
can look up synonyms, antonyms, defintions, and context
"""
syns = wordnet.synsets("program")

print(syns)

print(syns[0])

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definiton
print(syns[0].definition())

#examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        #print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


"""
semantic similarity
"""

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2) * 100)

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

print(w1.wup_similarity(w2) * 100)

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")

print(w1.wup_similarity(w2) * 100)

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cactus.n.01")

print(w1.wup_similarity(w2) * 100)

"""
Uses: 
test similarity of term papers
including using simlarity to test
if students are switching words
for synonyms but still plagiarizing
"""
