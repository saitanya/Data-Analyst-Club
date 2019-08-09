#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from data_processing import build_vocab, read_vocab, process_file_words

word_seq_length = 20
vocab_size = 10000 
base_dir = 'data'
train_dir = os.path.join(base_dir, 'Feature_train.txt')
test_dir = os.path.join(base_dir, 'Feature_test.txt')
val_dir = os.path.join(base_dir, 'Feature_validate.txt')
vocab_dir = os.path.join(base_dir, 'Feature_vocab.txt')

if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir,vocab_size)
words, word_to_id = read_vocab(vocab_dir)

vocab_size = len(words)
x_train, y_train = process_file_words(train_dir, word_to_id, word_seq_length)
x_val, y_val = process_file_words(val_dir, word_to_id, word_seq_length)
x_test, y_test = process_file_words(test_dir, word_to_id, word_seq_length)


print("Multinomial Naı̈ve Bayes")
model = MultinomialNB()
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("L1-Regularized Support Vector Machine")
model = SVC(gamma='auto')
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("L2-Regularized Support Vector Machine") #with linear kernel
model = LinearSVC(penalty='l2')
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("L1-Regularized Logistic Regression")
model = LogisticRegression(penalty='l1')
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("L2-Regularized Logistic Regression")
model = LogisticRegression(penalty='l2')
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("Random Forest")
model = RandomForestClassifier()
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))

print("Gradient Boosting Decision Tree")
model = GradientBoostingClassifier()
model.fit(x_train, y_train)
ynew = model.predict(x_test)
print("accuracy: ",accuracy_score(y_test, ynew))
print("precision: ",precision_score(y_test, ynew, pos_label='Y'))
print("recall: ",recall_score(y_test, ynew, pos_label='Y'))
print("Confusion Matrix\n", confusion_matrix(y_test, ynew))
