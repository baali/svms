# -*- coding: utf-8 -*-
# based on https://bitbucket.org/mimirtech/nltk-bayes-classifier
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from collections import defaultdict
from nltk.classify.util import accuracy
from nltk.metrics import precision, recall
import re

train_samples = {
    'I hate you and you are a bad person': 'neg',
    'I love you and you are a good person': 'pos',
    'I fail at everything and I want to kill people' : 'neg',
    'I win at everything and I want to love people' : 'pos',
    'sad are things are happening. fml' : 'neg',
    'good are things are happening. gbu' : 'pos',
    'I am so poor' : 'neg',
    'I am so rich' : 'pos',
    'I hate you mommy ! You are my terrible person' : 'neg',
    'I love you mommy ! You are my amazing person' : 'pos',
    'I want to kill butterflies since they make me sad' : 'neg',
    'I want to chase butterflies since they make me happy' : 'pos',
    'I want to hurt bunnies' : 'neg',
    'I want to hug bunnies' : 'pos',
    'You make me frown' : 'neg',
    'You make me smile' : 'pos',
}
 
test_samples = [
    ('You are a terrible person and everything you do is bad', 'neg'),
    ('I love you all and you make me happy', 'pos'),
    ('I frown whenever I see you in a poor state of mind', 'neg'),
    ('Finally getting rich from my ideas. They make me smile.', 'pos'),
    ('My mommy is poor', 'neg'),
    ('I love butterflies. Yay for happy', 'pos'),
    ('Everything is fail today and I hate stuff', 'neg'),
#    ('This was not good!', 'neg'),
]
 
def gen_bow(s):
    words = s.split(' ')
    bow = {}
    for word in words:
        bow[word] = True
    return bow

def get_labeled_features(samples):
    '''
    This function is to get number of times a feature occurs under a
    given label.
    '''
    word_freqs = {}
    for text, label in train_samples.items():
        # Removing punctuations, lower case.
        text = text.lower()
        text = re.sub(r"[!@#$%\^&\*()\[\]{};:<>.?/\|,\\]+"," ",text)
        text = re.sub(r"'\"","", text)
        text = text.strip()

        tokens = text.split()
        for token in tokens:
            if token not in word_freqs:
                word_freqs[token] = {'pos': 0, 'neg': 0}
            word_freqs[token][label] += 1
    return word_freqs

def get_label_probdist(labeled_features):
    '''
    This returns frequency distribution of all the labels for each
    feature. It is sum up of number of times a feature is found in a
    certain label.
    '''
    label_fd = FreqDist()
    for item,counts in labeled_features.items():
        for label in ['neg','pos']:
            label_fd.inc(label, counts[label])
    label_probdist = ELEProbDist(label_fd)
    return label_probdist

def get_feature_probdist(labeled_features, labeled_probdist):
    '''
    Creating probability distribution for each feature
    '''
    feature_freqdist = defaultdict(FreqDist)
    feature_values = defaultdict(set)
    for token, counts in labeled_features.items():
        for label in ['neg','pos']:
            feature_freqdist[label, token].inc(True, count=counts[label])
            feature_freqdist[label, token].inc(None, labeled_probdist._freqdist[label] - counts[label])
            feature_values[token].add(None)
            feature_values[token].add(True)
    feature_probdist = {}
    for ((label, fname), freqdist) in feature_freqdist.items():
        probdist = ELEProbDist(freqdist, bins=len(feature_values[fname]))
        feature_probdist[label,fname] = probdist
    return feature_probdist

labeled_features = get_labeled_features(train_samples)
label_probdist = get_label_probdist(labeled_features)
feature_probdist = get_feature_probdist(labeled_features, label_probdist)
classifier = NaiveBayesClassifier(label_probdist, feature_probdist)

if __name__ == '__main__':
    for sample, label in test_samples:
        print "%s | %s" % (sample, classifier.classify(gen_bow(sample)))
    classifier.show_most_informative_features()
    # getting accuracy
    test_samples = [(gen_bow(fs), l) for (fs, l) in test_samples]
    print 'accuracy for the system', accuracy(classifier, test_samples)
    # getting precision and recall stats
    refsets = defaultdict(set)
    testsets = defaultdict(set)
    for i, (feat, label) in enumerate(test_samples):
        refsets[label].add(i)
        observed = classifier.classify(feat)
        testsets[observed].add(i)
    labels = classifier.labels()
    for label in labels:
        ref = refsets[label]
        test = testsets[label]
        print '%s precision: %f' % (label, precision(ref, test) or 0)
        print '%s recall: %f' % (label, recall(ref, test) or 0)
