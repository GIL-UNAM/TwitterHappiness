#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemming = SnowballStemmer("spanish")
stops = set(stopwords.words('spanish'))
#from sklearn.feature_extraction.text import CountVectorizer


################################################# Definiciones #####################################################
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
	print(new_data)
	#new_data.to_csv('Dataset.csv', header=True, index=True)

