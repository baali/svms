# -*- coding: utf-8 -*-
# based on https://bitbucket.org/mimirtech/nltk-bayes-classifier
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import collections

train_samples = collections.OrderedDict({
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
    'You are a terrible person and everything you do is bad' : 'neg',
    'I love you all and you make me happy' : 'pos',
    'I frown whenever I see you in a poor state of mind' : 'neg',
    'Finally getting rich from my ideas. They make me smile.' : 'pos',
    'My mommy is poor' : 'neg',
    'I love butterflies. Yay for happy' : 'pos',
    'Everything is fail today and I hate stuff' : 'neg',
})
 
if __name__ == '__main__':
    # divide data in training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(
        train_samples.keys(), train_samples.values(), test_size=0.2, random_state=42)
    # tokenizing and creating sparse array from words of text
    cv = CountVectorizer()
    cv_arr_train = cv.fit_transform(X_train)
    cv_arr_test = cv.transform(X_test)
    # simple default classifier with no tuning
    clf = svm.SVC()
    # training classifier, or fitting the data
    clf.fit(cv_arr_train, y_train)
    print('Classifier performance with default parameters:')
    for index, sample in enumerate(X_test):
        # convert text into sparse array of same dimension as training
        # set
        sample_arr = cv.transform([sample])
        print("%s | %s" % (sample, clf.predict(sample_arr)))
    print('')
    score = clf.score(cv_arr_test, y_test)
    print("Score for default SVM %0.3f" %(score))
    # Tuning different parameters for best performance
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(svm.SVC(C=1, probability=True),
                       tuned_parameters, cv=5)
    # training classifier, or fitting the data
    clf.fit(cv_arr_train, y_train)
    print('Classifier performance after tuning parameters:')
    for index, sample in enumerate(X_test):
        # convert text into sparse array of same dimension as training
        # set
        sample_arr = cv.transform([sample])
        print "%s | %s" % (sample, clf.predict(sample_arr))
    score = clf.score(cv_arr_test, y_test)
    print("Score for tuned SVM %0.3f" %(score))
