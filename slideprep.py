from preprocessor import *

orgFile  = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextAll.txt'
prepFile = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/preprocessed_corpusfile_1120.txt'
category = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB_Catdictionary.txt'

orgdocument = list(read_input(orgFile))
prepdocument = list(read_input(prepFile))

orgvocabs = {}
for i, sentences in enumerate(orgdocument):
	for word in sentences.split(' '):
		if word not in orgvocabs:
			orgvocabs[word]=1
print("The original vocabulary size is {}.".format(len(orgvocabs)))

prepvocabs={}
for i, sentences in enumerate(prepdocument):
	for word in sentences.split(' '):
		if word not in prepvocabs:
			prepvocabs[word]=1
print("After pre-processing, the vocabulary size is {}.".format(len(prepvocabs)))

with open(category, 'rb') as f:
	catdict = pickle.load(f)
print(len(catdict))
print(catdict)