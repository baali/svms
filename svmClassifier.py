#!/usr/bin/python
# -*- coding: utf-8 -*-
#

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import os

if __name__ == '__main__':
    CURR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE_PATH = os.path.join(CURR_DIR, 'test_file')
    # complete category list is ['Antiques', 'Art', 'Baby', 'Books',
    # 'Crafts', 'Jewelry & Watches', 'Music', 'Stamps', 'Tickets',
    # 'Travel']
    # limiting data to just to categories for now.
    categories = ['Music',
                  'Stamps',
                  ]
    data = open(DATA_FILE_PATH).read().strip()
    lines = data.split('\n')
    sk_data = {}
    sk_data['target'] = []
    sk_data['data'] = []

    for line in lines:
        target, text = line.split('\t', 1)
        if target.strip() in categories:
            sk_data['target'].append(target.strip())
            sk_data['data'].append(text.strip())
    # divide data in training and testing samples            
    X_train, X_test, y_train, y_test = train_test_split(
        sk_data['data'], sk_data['target'], test_size=0.2, random_state=0)    

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
