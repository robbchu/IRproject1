import gensim
import gzip
import logging
from nltk import sent_tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpusFile = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/preprocessed_corpusfile_1120.txt.gz'

def read_input(input_file, sentence_tokenize=1):
	'''The method reads the input file in gzip format'''

	logging.info("reading file {0}...this may take a while".format(input_file))

	with gzip.open(corpusFile, 'r') as f:
		for i, line in enumerate(f):

			line = line.decode('utf-8')
			if i<5:
				print(line)
			if(i%10000==0):
				logging.info("read {0} lines".format(i))

			if(sentence_tokenize==1):
				for tok_sentence in sent_tokenize(line):
					yield gensim.utils.simple_preprocess(tok_sentence)
			else:
				yield gensim.utils.simple_preprocess(line)

document = list(read_input(corpusFile))
#print(document[0:3])
logging.info("Done reading data file")

model = gensim.models.Word2Vec(document, size=300, min_count=5, window=5, workers=4, sg=1)
model.train(document, total_examples=len(document), epochs=10)

vocabsize=0
for i, sentence in enumerate(document):
	vocabsize+=len(document[i])
print(vocabsize)
print(model.vector_size)
print(len(model.wv.vocab))
print(len(list(model.wv.vocab)))

model.save('QLforum_1120.bin')