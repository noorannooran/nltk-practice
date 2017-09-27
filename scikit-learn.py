import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

#one-liner collecting movie reviews into documents for training and testing sets
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#several liner / above deconstructed
##documents = []
##
##for category in movie_reviews.categories():
##    for fileid in movie_reviews.fileids(category):
##        documents.append(list(movie_reviews.words(fileid)), category)

random.shuffle(documents)

#print(documents[1])
#take all words, compile, find most popular words use, which in positive,
# which in negative, which one has more positive/ negative words = classified!

#adds all words to a list
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# convert to frequency distribution (includes punctuation)
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15)) #15 most common words
#print(all_words["stupid"])       #how many times word is used

#check the top 3000 words, keys only
word_features = list(all_words.keys())[:3000]

#quick function to find features within document
def find_features(document):
    words = set(document) #only one iteration of unique element
    features = {}
    for w in word_features:
        features[w] = (w in words) #boolean T or F

    return features


#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#load pickled classifier
classifier_f = open("naivebayes2.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

#Naive-Bayes algorithm (works on very strong independent functions) "Stupid Bayes"
#posterior (likelihood) = prior occurrences * likelihood / evidence

#create classifier
#train classifier against the training set
#classifier = nltk.NaiveBayesClassifier.train(training_set)
#show accuracy
print("Original Naive Bayes Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100
)
#show top 15 words and how many times they appear in pos/neg reviews
classifier.show_most_informative_features(15)

###save classifier with pickle - write as bytes
##save_classifier = open("naivebayes2.pickle", "wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()


#using scikit-learn module
#MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algo Accuracy Percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100
)

##TypeError (Sparse matrix was passed but dense matrix is required)
#GaussianNB
##GaussianNB_classifier = SklearnClassifier(GaussianNB())
##GaussianNB_classifier.train(training_set)
##print("Gaussian Naive Bayes Algo Accuracy Percent:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100
##)

#BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Bernoulli Naive Bayes Algo Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)

##LogisticRegression, SGDClassifier
##SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print("SGDClassifier Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("Linear SVC Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Accuracy Percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100
)
