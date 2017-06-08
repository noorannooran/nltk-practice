import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

#chunking: chunk into "noun phrases" for example (nouns with modifiers)
#can only chunk words that are 'touching'

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
        
            chunkGram = """Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""" #adverb, verb, proper noun, noun

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunkGram2 = """Chunk: {<NN.?>*<VB.?>*}"""
            chunkParser2 = nltk.RegexpParser(chunkGram2)
            chunked2 = chunkParser2.parse(tagged)

            chunked.draw()
            chunked2.draw()
            
    except Exception as e:
        print(str(e))
    

process_content()

