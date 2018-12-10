from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np

import spacy
spacynlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)

filename_CQA = 'QLforum_1120.bin'
filename_GNews='GoogleNews-vectors-negative300.bin'

model2test = Word2Vec.load(filename_CQA)
'''
Top5toMassage = model2test.wv.most_similar(positive='massage', topn=5)
top5=['massage']
for i, entity in enumerate(Top5toMassage):
	top5.append(entity[0])

X = model2test[top5]
pca = PCA(n_components=2)
pcaResult = pca.fit_transform(X)

pyplot.scatter(pcaResult[:, 0], pcaResult[:, 1])
for i, word in enumerate(top5):
	pyplot.annotate(word, xy=(pcaResult[i, 0], pcaResult[i, 1]))
'''


model2compare = KeyedVectors.load_word2vec_format(filename_GNews, binary=True)

Top5toMassage = model2compare.most_similar(positive='massage', topn=5)
top5=['massage']
for i, entity in enumerate(Top5toMassage):
	top5.append(entity[0])
print(top5)
'''
X = model2compare[top5]
pca = PCA(n_components=2)
pcaResult = pca.fit_transform(X)

pyplot.scatter(pcaResult[:, 0], pcaResult[:, 1])
for i, word in enumerate(top5):
	pyplot.annotate(word, xy=(pcaResult[i, 0], pcaResult[i, 1]))

pyplot.show()
'''
'''
X = model2test[model2test.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
#Create a scatter plot of the projection
pyplot.scatter(result[0:30, 0], result[0:30, 1])
words = list(model2test.wv.vocab)[0:30]
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
'''

'''
sentence = "cream color building with light blue glass window"

sentence = spacynlp(sentence)
for i, word in enumerate(sentence):
	if(word.pos_=="NOUN"):
		print(word)
'''
'''
wordvecs=[]
for word in sentence.split(' '):
	if word in model2test.wv.vocab:
		wordvecs.append(model2test[word])
	else:
		wordvecs.append([0]*128)
print(len(wordvecs))

print(model2test['good'])
print(model2test['smelly'].shape)
print((model2test['good']+model2test['smelly']+np.zeros(model2test.vector_size, dtype=np.float))/2)
'''

#Fit a 2d PCA model to the vectors

