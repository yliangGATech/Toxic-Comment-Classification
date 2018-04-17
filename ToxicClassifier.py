
# coding: utf-8
"""
Data Visualizations for text comments:
scatter plot
distributions

"""
# In[ ]:

import sklearn
import numpy as np
from numpy import random
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import time


# In[2]:

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[ ]:

training_data = 'TrainingData.xlsx'
testing_data = 'TestingData.xlsx'
train_xl = pd.ExcelFile(training_data)
sheet_names = train_xl.sheet_names
sheet_names[0]
df = train_xl.parse('train')
df.to_pickle('training_data.pkl')
df.iloc[1:15]


# In[ ]:

dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[1,1]


# In[4]:

train_df = pd.read_pickle('training_data.pkl')
#train_df.iloc[1:15]
toxic_col = train_df['toxic']
#toxic_col[1:15]
toxic_comments = train_df.loc[toxic_col == 1]
#toxic_comments[1:15]
non_toxic_comments = train_df.loc[toxic_col == 0]
#non_toxic_comments[1:15]
toxic_comments.to_pickle('toxic_comments.pkl')
non_toxic_comments.to_pickle('non_toxic_comments')


# In[10]:

all_words = []
documents = []
allowed_word_types = ["JJ","JJR","JJS","NN", "NNS","RB","RBR","VB","VBD", "VBG", "VBN", "VBP", "VBZ"]
i = 0
for p in toxic_comments['comment_text']:
    if type(p) is not str:
        continue
    if i > 10000:
        break
    documents.append( (p, "tox") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    if (i % 1000 == 0):
        print(i / 1000)
    i = i + 1
    for w in pos:
        if w[1] in allowed_word_types:
            all_words.append(w[0].lower())

i = 0
for p in non_toxic_comments['comment_text']:
    if type(p) is not str:
        continue
    if i > 10000:
        break
    documents.append( (p, "cln") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    if (i % 1000 == 0):
        print(i / 1000)
    i = i + 1
    for w in pos:
        if w[1] in allowed_word_types:
            all_words.append(w[0].lower())




# In[17]:




# In[20]:

save_documents = open("labeled_data/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:10000]



save_word_features = open("labeled_data/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



# In[ ]:




# In[21]:

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
print('complete')


# In[ ]:

documents[:15]


# In[ ]:

#featuresets = [(find_features(comment), toxicity) for (comment, toxicity) in documents]
featuresets = []
i = 0
for tup in documents:
    if i % 1000 == 0:
        print(i)
    i = i + 1
    try:
        featuresets.append((find_features(tup[0]), tup[1]))
    except:
        print(tup[0])
print('complete')


random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

###############
save_classifier = open("classifiers/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# In[ ]:

print(training_set[1:15])
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("classifiers/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("classifiers/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("classifiers/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("classifers/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("classifiers/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()


# In[ ]:




# In[ ]:




# In[ ]:



