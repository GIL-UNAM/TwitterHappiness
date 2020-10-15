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

################################################# Funciones #####################################################
def rowse(DataFrame):
	return DataFrame.iloc[0:10048, 0]
def rowst(DataFrame):
	return DataFrame.iloc[0:10048, 1]


def apply_cleaning_function_to_list(X):
	cleaned_X = []
	for x in X:
		cleaned_X.append(clean_text(x))
	return cleaned_X
def clean_text(raw_text):
	text = raw_text.lower()
	tokens = nltk.word_tokenize(text)
	token_words = [w for w in tokens if w.isalpha()]
	gt = [w for w in tokens if w == 'gt']
	punct = [re.sub(r'[^\w\s]', '', w) for w in token_words if not w in gt]
	stemmed_words = [stemming.stem(w) for w in punct]
	meaningful_words = [w for w in stemmed_words]
	
	joined_words = (' '.join(meaningful_words))
	return joined_words

#################################################################################################################

if __name__ == "__main__":

	text1 = pd.read_csv('alegria_claudia_terminado.csv', names = ['Etiqueta1', 'Tweet1'])
	text1['Tweet1'] = text1['Tweet1'].replace('http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

	text2 = pd.read_csv('alegria_tona_terminado.csv', names = ['Etiqueta2', 'Tweet2'])
	text2['Tweet2'] = text2['Tweet2'].replace('http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

	text3 = pd.read_csv('alegria_iris.csv', names = ['Etiqueta3', 'Tweet3'])
	text3['Tweet3'] = text3['Tweet3'].replace('http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

	tweets = pd.concat([rowst(text1), rowst(text2), rowst(text3)])

	etiquetas = pd.concat([rowse(text1), rowse(text2), rowse(text3)])
	etiquetas = etiquetas.replace(' N', 'N')
	etiquetas = etiquetas.replace('N ', 'N')
	etiquetas = etiquetas.replace('X', 'C')
	etiquetas = etiquetas.replace('V', 'C')
	etiquetas = etiquetas.replace('S', 'A')
	etiquetas = etiquetas.replace('D', 'A')
	etiquetas = etiquetas.replace('O', 'P')
	etiquetas = etiquetas.replace(' F', 'F')
	etiquetas = etiquetas.replace('C ', 'C')
	etiquetas = etiquetas.replace('Á', 'A')

	data = pd.DataFrame({'et1': rowse(text1), 'et2': rowse(text2),
		'et3': rowse(text3) , 'twits': rowst(text1)})
	data['et1'] = np.where(data['et1'] == 'Á', 'A', data['et1'])
	data['et2'] = np.where(data['et2'] == 'X', 'C', data['et2'])
	data['et2'] = np.where(data['et2'] == 'V', 'C', data['et2'])
	data['et2'] = np.where(data['et2'] == 'S', 'A', data['et2'])
	data['et2'] = np.where(data['et2'] == 'D', 'A', data['et2'])
	data['et2'] = np.where(data['et2'] == ' N', 'N', data['et2'])
	data['et3'] = np.where(data['et3'] == 'N ', 'N', data['et3'])
	data['et3'] = np.where(data['et3'] == 'O', 'P', data['et3'])
	data['et3'] = np.where(data['et3'] == ' F', 'F', data['et3'])
	data['et3'] = np.where(data['et3'] == 'C ', 'C', data['et3'])
	data['et3'] = np.where(data['et3'] == 'V', 'C', data['et3'])

	et1 = list(enumerate(data['et1']))
	et2 = list(enumerate(data['et2']))
	et3 = list(enumerate(data['et3']))

	atl2 = set(et1).intersection(et2).union(set(et1).intersection(et3)).union(set(et2).intersection(et3))
	ex3 = set(et1).intersection(et2).intersection(set(et1).intersection(et3))

	union = list(sorted(atl2.union(ex3)))

	index = [x for x,y in et1]
	iunion = [x for x,y in union]
	eunion = [y for x,y in union]

	etiq = pd.Series(eunion, index=iunion)
	data['match'] = etiq.reindex()
	data = data.fillna('NA')


		
	text_to_clean = list(data['twits'])
	cleaned_text = apply_cleaning_function_to_list(text_to_clean)
	data['cleaned'] = cleaned_text
	new_data = pd.DataFrame({'text': data.cleaned, 'target': data.match})
	new_data = new_data[new_data.target != 'NA'] 	#Se eliminan los que quedaron sin categoría
	#print(new_data)


	X = new_data['text']
	y = new_data['target']

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

