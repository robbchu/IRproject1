from __future__ import unicode_literals, print_function, division
from io import open
from gensim.models import Word2Vec 
import unicodedata
import string
import re
import random
import itertools
import numpy as np

import pickle
import spacy
import nltk

word2vec = Word2Vec.load('QLforum_1120.bin')
wv_dim = word2vec.vector_size#300
vocabsize = len(word2vec.wv.vocab)

spacynlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
#Read in the corpus text pair for modeling
file2read1 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB_prep.txt"
file2read2 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/test_data/testPair.txt"
'''
The Pair Format
[(0,1,2)ORGQ_ID, OrgQSubject.text, OrgQBody.text,\
 (3,4,5)RELQ_ID, RELQ_RANKING_ORDER, RELQ_CATEGORY,\
 (6,7,8)RelQSubject.text, RelQBody.text, RELQ_RELEVANCE2ORGQ]
'''
#To read the LIST object back
with open(file2read1, 'rb') as f2r1:
	pairList = pickle.load(f2r1)
with open(file2read2, 'rb') as f2r2:
	testList = pickle.load(f2r2)
#Record the pair features
relcount=1530
nonrelcount=2339
maxlength=500
minlength=25

average_wordinrelq = 28
std_wordinrelq = 13
realtedQ=relatedQ_wordinsentence=0
maxrelqlen=85
minrelqlen=2
maxorgqlen=63
minorgqlen=2

relatedQ_countlist=[]
#start process the pairList for the subtaskB

'''Prepare data in the form of posPair, negPair, word2index...etc'''
SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 10

'''posPair for the relevant orgq/relq pairs, vice versa for irrelevant pairs'''
def preparePairData(word2index={}, word2count={}, index2word={0:"SOS", 1:"EOS", 2:"UNK"}, n_words=3, word2vec_used=1):
	posPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in pairList \
			if (pair[8]=="PerfectMatch" or pair[8]=="Relevant")]
	negPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in pairList \
			if (pair[8]=="Irrelevant")]
	testPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in testList]
	
	return (posPairs, negPairs, testPairs)
#print(random.choice(testPairs), len(testPairs))

def wordvecFromToken(token):
	if token in word2vec.wv.vocab:
		return (word2vec[token])
	else:
		return (np.zeros(wv_dim, dtype=np.float64))

def wordvecFromSentence(sentence):
	wordvecs = []
	for word in sentence.split(' '):
		if word in word2vec.wv.vocab:
			wordvecs.append(word2vec[word].tolist())
		else:
			wordvecs.append([0.0]*wv_dim)
	return wordvecs

def evaluate(pairs):
	now_orgqID = testList[0][0]
	total_score=[]
	scores=[]
	for i, pair in enumerate(pairs):
		if(testList[i][0]!=now_orgqID):
			now_orgqID = testList[i][0]
			total_score.append(scores)
			scores=[]

		orgq = pair[0]
		relq = pair[1]

		orgq_nlp = spacynlp(orgq)
		relq_nlp = spacynlp(relq)

		orgq_wvlist = np.zeros(wv_dim, dtype = np.float64)
		orgq_wvcount=0
		for j, word in enumerate(orgq_nlp):
			if(word.pos_=="NOUN"):
				orgq_wvlist += np.asarray(wordvecFromToken(word.text))
				orgq_wvcount+=1
		relq_wvlist = np.zeros(wv_dim, dtype = np.float64)
		relq_wvcount=0
		for j, word in enumerate(relq_nlp):
			if(word.pos_=="NOUN"):
				relq_wvlist += np.asarray(wordvecFromToken(word.text))
				relq_wvcount+=1
		
		orgq_wvlist /= (orgq_wvcount+1) 
		relq_wvlist /= (relq_wvcount+1)

		similarity = np.sum(np.multiply(orgq_wvlist, relq_wvlist))
		scores.append(similarity)

	total_score.append(scores)

	return total_score

def predict(total_score):
	prediction=""

	for i, scores in enumerate(total_score):
		mean_score = np.mean(np.asarray(scores))
		std_score  = np.std(np.asarray(scores))
		for j, score in enumerate(scores):
			if(score>mean_score+std_score):
				label="true"
			else:
				label="false"
			prediction+=\
				testList[i*len(scores)+j][0]+"\t"+testList[i*len(scores)+j][3]+"\t"+"0"+"\t"+str(score)+"\t"+label+"\n"

	return prediction

if __name__ == '__main__':
	posPairs, negPairs, testPairs = preparePairData()
	prediction = predict(evaluate(testPairs))
	file2write1 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/relevancyB_wv.txt"
	with open(file2write1, 'w') as f2w1:
		f2w1.write(prediction)