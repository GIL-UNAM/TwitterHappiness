#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from collections import Counter
import re
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemming = SnowballStemmer("spanish")
stops = set(stopwords.words('spanish'))
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


if __name__ == "__main__":

	dataset = pd.read_csv('Dataset.csv')

	X = dataset['text']
	y = dataset['target']

	skf = StratifiedKFold(n_splits=3)
	for train_index,test_index in skf.split(X,y):
		print ("entrenamiento:",len(train_index),"prueba",len(test_index))
			
		X_train,X_test = X.iloc[train_index],X.iloc[test_index]
		y_train,y_test = y.iloc[train_index],y.iloc[test_index]

		vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))#, use_idf=False)
		tfidf = TfidfTransformer()
		train_vectors = vectorizer.fit_transform(X_train)
		train_data_tfidf_features = tfidf.fit_transform(train_vectors)

		test_vectors = vectorizer.transform(X_test)
		test_data_features = test_vectors.toarray()
		test_data_tfidf_features = tfidf.fit_transform(test_data_features)
		test_data_tfidf_features = test_data_tfidf_features.toarray()


		print('NB...')
		Naive = naive_bayes.MultinomialNB()
		Naive.fit(train_data_tfidf_features,y_train)
		predicted_y = Naive.predict(test_data_tfidf_features)
		correctly_identified_y = predicted_y == y_test
		accuracy = np.mean(correctly_identified_y) * 100
		print ('Accuracy = %.0f%%' %accuracy)


		print('LR...')
		ml_model = LogisticRegression(solver='liblinear', multi_class='auto', C = 100,random_state = 0)
		ml_model.fit(train_data_tfidf_features,y_train)
		predicted_y = ml_model.predict(test_data_tfidf_features)
		correctly_identified_y = predicted_y == y_test
		accuracy = np.mean(correctly_identified_y) * 100
		print ('Accuracy = %.0f%%' %accuracy)


		print('RF...')
		clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
		clf.fit(train_data_tfidf_features,y_train)
		predicted_y = clf.predict(test_data_tfidf_features)
		correctly_identified_y = predicted_y == y_test
		accuracy = np.mean(correctly_identified_y) * 100
		print ('Accuracy = %.0f%%' %accuracy)


		print('SVM...')
		model = LinearSVC(tol=1.0e-6, max_iter=5000, verbose=0)
		model.fit(train_data_tfidf_features,y_train)
		predicted_y = model.predict(test_data_tfidf_features)
		correctly_identified_y = predicted_y == y_test
		accuracy = np.mean(correctly_identified_y) * 100
		print ('Accuracy = %.0f%%' %accuracy)


	print('NB...')
	Naive = naive_bayes.MultinomialNB()
	Naive.fit(train_data_tfidf_features,y_train)
	scoresNB = cross_val_score(Naive, train_vectors,y_train, cv=skf)
	print(scoresNB)
	print("Scores mean: %.2f%%" % (scoresNB.mean()*100.0))
		
	print('LR...')
	ml_model = LogisticRegression(solver='liblinear', multi_class='auto',
		C = 100,random_state = 0)
	ml_model.fit(train_data_tfidf_features,y_train)
	scoresLR = cross_val_score(ml_model, train_vectors,y_train, cv=skf)
	print(scoresLR)
	print("Scores mean: %.2f%%" % (scoresLR.mean()*100.0))
			
	print('RF...')
	clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
	clf.fit(train_data_tfidf_features,y_train)
	scoresRF = cross_val_score(clf, train_vectors,y_train, cv=skf)
	print(scoresRF)
	print("Scores mean: %.2f%%" % (scoresRF.mean()*100.0))

	print('SVM...')
	model = LinearSVC(tol=1.0e-6, max_iter=5000, verbose=0)
	model.fit(train_data_tfidf_features,y_train)
	scoresSVM = cross_val_score(model, train_vectors,y_train, cv=skf)
	print(scoresSVM)
	print("Scores mean: %.2f%%" % (scoresSVM.mean()*100.0))

