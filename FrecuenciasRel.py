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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sn

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
	punct = [re.sub(r'[^\w\s]', '', w) for w in token_words if not w in stops]
	#stemmed_words = [stemming.stem(w) for w in punct]
	#meaningful_words = [w for w in stemmed_words if not w in stops]
	#joined_words = (' '.join(meaningful_words))
	joined_words = (' '.join(punct))
	return joined_words

def percent(x):
	y = ((x)*100)/10048
	return float("{0:.2f}".format(y))

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


	############################################################# CANTIDAD DE TUITS Y PORCENTAJES ############################################################################

	cant = list(data.groupby('match').twits.count().values)
	categ = sorted(list(data.match.unique()))
	percents = [percent(x) for x in cant]
	
	info = pd.DataFrame(data.groupby('match').twits.count())
	info['percent'] = percents
	
	#for x,y,z in zip(categ,cant,percents):
		#print('Categoría: {} cantidad de tuits: {} porcentaje: {}%'.format(x,y,z))

	############ Gráfica de barras
	fig = plt.figure(figsize=(8,6))
	bar = data.groupby('match').twits.count().plot.bar (ylim=0)
	plt.xlabel('Categorías') 
	plt.ylabel('Cantidad de tuits')
	plt.grid(alpha=.5, linestyle='--')
	plt.show()


	########################################################### FRECUENCIA DE PALABRA EN CADA CATEGORÍA ######################################################################

	text_to_clean = list(data['twits'])
	cleaned_text = apply_cleaning_function_to_list(text_to_clean)
	data['cleaned'] = cleaned_text
	new_data = pd.DataFrame({'text': data.cleaned, 'target': data.match})
	new_data = new_data[new_data.target != 'NA'] 	#Se eliminan los que quedaron sin categoría
	#print(new_data)

	clasA = new_data[new_data.target == 'A']
	clasP = new_data[new_data.target == 'P']
	clasF = new_data[new_data.target == 'F']
	clasC = new_data[new_data.target == 'C']
	clasN = new_data[new_data.target == 'N']
	#print(new_data)

	cvec = CountVectorizer(analyzer = 'word')
	cvec.fit(new_data.text)

	clasA_matrix = cvec.transform(clasA.text)
	clasP_matrix = cvec.transform(clasP.text)
	clasF_matrix = cvec.transform(clasF.text)
	clasC_matrix = cvec.transform(clasC.text)
	clasN_matrix = cvec.transform(clasN.text)

	A_tf = np.sum(clasA_matrix,axis=0)
	P_tf = np.sum(clasP_matrix,axis=0)
	F_tf = np.sum(clasF_matrix,axis=0)
	C_tf = np.sum(clasC_matrix,axis=0)
	N_tf = np.sum(clasN_matrix,axis=0)
	
	cA = np.squeeze(np.asarray(A_tf))
	cP = np.squeeze(np.asarray(P_tf))
	cF = np.squeeze(np.asarray(F_tf))
	cC = np.squeeze(np.asarray(C_tf))
	cN = np.squeeze(np.asarray(N_tf))
	
	term_freq_df = pd.DataFrame([cA,cP,cF,cC,cN],
		columns=cvec.get_feature_names()).transpose()
	term_freq_df.columns = ['A','P','F','C','N']
	term_freq_df['total'] = (term_freq_df['A'] + term_freq_df['P'] + term_freq_df['F']
		+ term_freq_df['C'] + term_freq_df['N'])
	#print(term_freq_df.sort_values(by='total', ascending=False))
	

	################# Frecuencia relativa
	frec_rel = pd.DataFrame()
	frec_rel['A'] = term_freq_df['A'].apply(lambda x: (x/40351)*100)
	frec_rel['P'] = term_freq_df['P'].apply(lambda x: (x/8654)*100)
	frec_rel['F'] = term_freq_df['F'].apply(lambda x: (x/4332)*100)
	frec_rel['C'] = term_freq_df['C'].apply(lambda x: (x/21151)*100)
	frec_rel['N'] = term_freq_df['N'].apply(lambda x: (x/9279)*100)
	frec_rel['total'] = term_freq_df['total'].apply(lambda x: (x/97095)*100)
	#print(frec_rel.sort_values(by='total', ascending=False).head(20))


