import logging
import gzip
import gensim
import nltk
import spacy
import re
import unicodedata
import pickle

from nltk.tokenize.toktok import ToktokTokenizer
from nltk import sent_tokenize
from contractions import CONTRACTION_MAP

#nlp = spacy.load('en_core')
LanguageModel = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)

corpusFile = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextAll.txt'

def read_input(input_file):
	'''The method reads the input file in gzip format'''

	logging.info("reading file {0}...this may take a while".format(input_file))

	with open(input_file, 'rb') as f:
		for i, line in enumerate(f):

			if(i%10000==0):
				logging.info("read {0} lines".format(i))
			yield line.decode('utf-8')

def lowercase(sentence):

	return sentence.lower()

def remove_extra_newline(sentence):
	sentence = re.sub(r'[\r|\b|\r\n]+', ' ', sentence)

	return sentence

'''Remove accented notation'''
def remove_accented_chars(sentence):
	sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		
	return sentence

'''Expand the contractions'''
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
	
	contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
									  flags=re.IGNORECASE|re.DOTALL)
	def expand_match(contraction):
		match = contraction.group(0)
		first_char = match[0]
		expanded_contraction = contraction_mapping.get(match)\
								if contraction_mapping.get(match)\
								else contraction_mapping.get(match.lower())                       
		expanded_contraction = first_char+expanded_contraction[1:]
		return expanded_contraction
		
	expanded_text = contractions_pattern.sub(expand_match, text)
	expanded_text = re.sub("'", "", expanded_text)
	return expanded_text

def expand_contractions_doc(document):
	expanded_document=[]
	for sentence in document:
		expanded_document.append(expand_contractions(sentence))
	return expanded_document

'''Remove special characters'''
def remove_special_characters(sentence, remove_digits=False):
	pattern = r'[^a-zA-z0-9\.\s]' if not remove_digits else r'[^a-zA-z\.\s]'
	sentence = re.sub(pattern, ' ', sentence)

	return sentence

#print(remove_special_characters(document, remove_digits=True)[0:5])

def simple_stemmer(sentence):
	ps = nltk.stem.PorterStemmer()
	sentence = ' '.join([ps.stem(word) for word in sentence.split()])

	return sentence

def lemmatize_document(sentence):
	sentence = LanguageModel(sentence)
	sentence = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in sentence])

	return sentence

tokenizer= ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
def remove_stopwords(sentence):
	tokens = tokenizer.tokenize(sentence)
	tokens = [token.strip() for token in tokens]
	filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
	filtered_sentence = ' '.join(filtered_tokens)

	return filtered_sentence


if __name__=='__main__':
	'''
	preprocessed_document = remove_special_characters(remove_stopwords(lemmatize_document(lowercase(expand_contractions_doc(remove_accented_chars(document[0:5]))))))
	for i, sentence in enumerate(document[0:5]):
		print(sentence)
		print(lemmatize_document(remove_extra_newline(document[0:5]))[i])
		print(preprocessed_document[i])
	'''

	document = list(read_input(corpusFile))
	print(len(document))


	preprocessed_document = []
	for i, sentences in enumerate(document):
		if i%10000==0:
			print('read {0} lines of document'.format(i))
		#for tok_sentence in sent_tokenize(sentences):
		preprocessed_sentence = remove_stopwords(remove_special_characters(lemmatize_document(lowercase(expand_contractions(remove_accented_chars(sentences))))))
		#print(tok_sentence)
		preprocessed_document.append(preprocessed_sentence)
	#print(preprocessed_document[0:5])


	#print(remove_accented_chars(document)[0:2])
	#print(expand_contractions_doc(remove_accented_chars(document))[0:2])
	#print(remove_stopwords(remove_special_characters(expand_contractions_doc(remove_accented_chars(document))))[0:2])

	#preprocessed_document = remove_stopwords(remove_special_characters(lemmatize_document(lowercase(expand_contractions_doc(remove_accented_chars(document))))))
	#preprocessed_document = remove_stopwords(remove_special_characters(expand_contractions_doc(remove_accented_chars(LowerCase(document)))))
	#print(len(preprocessed_document))
	filetowrite = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/preprocessed_corpusfile_1130.txt"
	with open(filetowrite, 'w') as f2w:
		for sentence in preprocessed_document:
			f2w.write(sentence+"\n")


	'''
	#ALSO preprocess the text pair for subtask		
	subtaskBPair = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB.txt"
	with open(subtaskBPair, 'rb') as f2r:
		pairList = pickle.load(f2r)
	for i, pair in enumerate(pairList):
		pair[1] = remove_stopwords(remove_special_characters(lemmatize_document\
			(lowercase(expand_contractions(remove_accented_chars(pair[1]))))))
		pair[2] = remove_stopwords(remove_special_characters(lemmatize_document\
			(lowercase(expand_contractions(remove_accented_chars(pair[2]))))))
		pair[6] = remove_stopwords(remove_special_characters(lemmatize_document\
			(lowercase(expand_contractions(remove_accented_chars(pair[6]))))))
		pair[7] = remove_stopwords(remove_special_characters(lemmatize_document\
			(lowercase(expand_contractions(remove_accented_chars(pair[7]))))))
	subtaskBPair_prep = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB_prep.txt"
	with open(subtaskBPair_prep, 'wb') as f2w2:
		pickle.dump(pairList, f2w2)
	'''