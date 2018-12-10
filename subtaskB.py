#Try singel push of branch pytorchtry
<<<<<<< HEAD
#branch of pytorchtry 2 master
=======
#branch of pytorchtry test in pytorchtry branch
>>>>>>> pytorchtry
from __future__ import unicode_literals, print_function, division
from io import open
from gensim.models import Word2Vec 
import unicodedata
import string
import re
import random
import itertools
import numpy as np

#another fix in pytorchtry branch
#another fix 2 in pytorch branch after merged from master

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pickle
import spacy
import nltk

word2vec = Word2Vec.load('QLforum_1120.bin')
wv_dim = word2vec.vector_size#128
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
def prepareData(word2index={}, word2count={}, index2word={0:"SOS", 1:"EOS", 2:"UNK"}, n_words=3, word2vec_used=1):
	posPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in pairList \
			if (pair[8]=="PerfectMatch" or pair[8]=="Relevant")]
	negPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in pairList \
			if (pair[8]=="Irrelevant")]
	testPairs=[[(pair[1]+" "+pair[2]), (pair[6]+" "+pair[7])] for pair in testList]
	if(word2vec_used==0):
		for pair in itertools.chain(posPairs, negPairs, testPairs):
			for word in itertools.chain(pair[0].split(' '), pair[1].split(' ')):
				if word not in word2index:
					word2index[word] = n_words
					word2count[word] = 1
					index2word[n_words] = word
					n_words += 1
				else:
					word2count[word] += 1
		for key, value in word2count.items():
			if(word2count[key]>5):
				continue
			else:
				word2index[key] = UNK_token
				n_words-=1
	
	return (posPairs, negPairs, testPairs, word2index, word2count, index2word, n_words)

posPairs, negPairs, testPairs, word2index, word2count, index2word, n_words = prepareData()
lenofpos = len(posPairs)
lenofneg = len(negPairs)
lenoftest= len(testPairs)
#print(random.choice(testPairs), len(testPairs))

'''Prepare training data'''
def indexesFromSentence(sentence):
	return [word2index[word] for word in (sentence.split(' ')[:MAX_LENGTH-1])]

def tensorFromSentence(sentence):
	indexes = indexesFromSentence(sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
	input_tensor = tensorFromSentence(pair[0])
	target_tensor = tensorFromSentence(pair[1])
	return (input_tensor, target_tensor)

def wvFromSentence_wv(sentence):
	wordvecs=[]
	for word in sentence.split(' ')[:MAX_LENGTH]:
		if word in word2vec.wv.vocab:
			wordvecs.append(word2vec[word].tolist())
		else:
			wordvecs.append([0.0]*wv_dim)
	return wordvecs

def tensorFromSentence_wv(sentence):
	wordvecs=[]
	for word in sentence.split(' ')[:MAX_LENGTH]:
		if word in word2vec.wv.vocab:
			wordvecs.append(word2vec[word].tolist())
		else:
			wordvecs.append([0.0]*wv_dim)
	return torch.tensor(wordvecs, device=device).view(len(wordvecs), wv_dim)

def tensorsFromPair_wv(pair):#return(input_tensor, target_tensor)
	return(tensorFromSentence_wv(pair[0]), tensorFromSentence_wv(pair[1]))

'''Seq2Seq'''
class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderRNN_wv(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN_wv, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		#output = self.embedding(input).view(1,1,-1)
		output = input.view(1,1,wv_dim)
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN_wv(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN_wv, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size*2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):#encoder_outputs[10,128]
		#embedded = self.embedding(input).view(1,1,-1)
		#embedded = self.dropout(embedded)
		output = input.view(1,1,wv_dim)#[1,1,128]
		#output = self.dropout(output)
		#bug#
		concat = torch.cat((output[0], hidden[0]), 1)#[1,256]
		preattn_weights = self.attn(concat)#[1,10]
		attn_weights = F.softmax(preattn_weights, dim=1)#[1,10]

		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))#[1,1,128]=bmm([1,1,10],[1,10,128])

		output = torch.cat((output[0], attn_applied[0]), 1)#[1, 256]
		
		output = self.attn_combine(output).unsqueeze(0)#[1, 1, 128]
		output= F.relu(output)#[1, 1, 128]
	
		output, hidden = self.gru(output, hidden)#input is (seq_len, batch_size, input_size)#[1, 1, 128]
		output = F.log_softmax(self.out(output[0]), dim=1)#[1, 128]

		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

def train_wv(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, \
	criterion, max_length=MAX_LENGTH):
	encoder_optim.zero_grad()
	decoder_optim.zero_grad()

	encoder_hidden = encoder.initHidden()
	#bug#
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	loss=0

	for ei in range(input_length):
		if(ei==10000):
				print("train", input_tensor[ei].type(), encoder_hidden.type())
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0,0]
		if(ei==10000):
			print("encoder output",encoder_output, encoder_output.size(), encoder_output[0,0], encoder_output.type())

	#bug#
	decoder_input = torch.tensor([[0.0]*wv_dim], device=device)
	decoder_hidden = encoder_hidden

	for di in range(target_length):
		decoder_output, decoder_hidden, decoder_attention = \
			decoder(decoder_input, decoder_hidden, encoder_outputs)
		if(di==10000):
			print("decoder output",decoder_output, decoder_output.size(), decoder_output.type())
			print("decoder hidden",decoder_hidden, decoder_hidden.size(), decoder_hidden.type())
			print("target tensor", target_tensor[di].view(1,-1), target_tensor[di].size())
		loss += criterion(decoder_output, target_tensor[di].view(1,-1))

		decoder_input = target_tensor[di]

	loss.backward()

	encoder_optim.step()
	decoder_optim.step()

	return loss.item() / target_length

import time
import math

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters_wv(encoder, decoder, n_iters, Pairs, learning_rate=0.01, print_every=100):
	encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair_wv(Pairs[i]) for i in range(n_iters)]

	#print(training_pairs[0][0].size())

	criterion = nn.MSELoss()

	start = time.time()
	print_loss_total=0
	for iter in range(1, n_iters+1):
		training_pair = training_pairs[iter-1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train_wv(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion)
		print_loss_total+=loss

		if iter % print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0
				print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),\
											 iter, iter / n_iters * 100, print_loss_avg))

hidden_size = 128
encoder1 = EncoderRNN_wv(wv_dim, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN_wv(hidden_size, wv_dim, dropout_p=0.1).to(device)
encoder2 = EncoderRNN_wv(wv_dim, hidden_size).to(device)
attn_decoder2 = AttnDecoderRNN_wv(hidden_size, wv_dim, dropout_p=0.1).to(device)

epochs=1
for epoch in range(epochs):
	print("start {}th epoch".format(epoch))
	trainIters_wv(encoder1, attn_decoder1, lenofpos, posPairs)
	trainIters_wv(encoder2, attn_decoder2, lenofneg, negPairs)

#Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
	with torch.no_grad():
		input_tensor = tensorFromSentence_wv(sentence)
		input_length = input_tensor.size(0)
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			if(ei==10000):
				print("eval", input_tensor[ei].type(), encoder_hidden.type())
				print(input_tensor[ei])
			encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		#decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
		decoder_input = torch.tensor([[0.0]*wv_dim], device=device)

		decoder_hidden = encoder_hidden

		decoded_words = []
		decoded_vectors = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = \
				decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
			#topv, topi = decoder_output.data.topk(1)
			#print(np.asarray(decoder_output.squeeze(0).data.tolist()))
			predword = word2vec.most_similar(positive=[np.asarray(decoder_output.squeeze(0).data.tolist())], topn=1)
			#print(predword[0][0])
			#print(decoder_output.squeeze(0).data.tolist())
			decoded_words.append(predword[0][0])
			decoded_vectors.append(decoder_output.squeeze(0).data.tolist())
			'''
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_index.append(topi.item())
				decoded_words.append(index2word[topi.item()])
			'''
			decoder_input = decoder_output

		return decoded_words, decoded_vectors, decoder_attentions[:di + 1]

def evaluateTestPair(encoder1, decoder1, encoder2, decoder2):
	#testPairTensor = [tensorsFromPair(pair) for pair in testPairs]
	prediction=""
	predword1=""
	predword2=""
	for i in range(len(testPairs)):
		output_words1, output_vectors1, attentions1 = evaluate(encoder1, decoder1, testPairs[i][0])
		output_words2, output_vectors2, attentions2 = evaluate(encoder2, decoder2, testPairs[i][0])
		
		output_sentence1 = ' '.join(output_words1)
		output_sentence2 = ' '.join(output_words2)
		predword1+=output_sentence1+"\n"
		predword2+=output_sentence2+"\n"
		#output_index1 = indexesFromSentence(output_sentence1)
		#output_index2 = indexesFromSentence(output_sentence2)
		#target_index  = indexesFromSentence(testPairs[i][1])
		target_vectors = wvFromSentence_wv(testPairs[i][1])
		if(i==1):
			print(len(output_vectors1))

		score1=0
		score2=0
		for j, wordvec in enumerate((target_vectors)):
			if (j<len(output_vectors1)):
				score1 += wv_dim**2 - (np.sum((np.array(output_vectors1[j])-np.array(wordvec))**2))
			if (j<len(output_vectors2)):
				score2 += wv_dim**2 - (np.sum((np.array(output_vectors2[j])-np.array(wordvec))**2))
		score1 = (score1/(len(target_vectors)+1))
		score2 = (score2/(len(target_vectors)+1))

		if(score1>score2):
			label = "true"
		else:
			label = "false"

		prediction += testList[i][0]+"\t"+testList[i][3]+"\t"+"0"+"\t"+str(score1)+"\t"+label+"\n"

	return prediction, predword1, predword2

prediction, predword1, predword2 = evaluateTestPair(encoder1, attn_decoder1, encoder2, attn_decoder2)

file2write1 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/relevancyB.txt"
with open(file2write1, 'w') as f2w1:
	f2w1.write(prediction)
file2write2 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/wordpred1B.txt"
with open(file2write2, 'w') as f2w2:
	f2w2.write(predword1)
file2write3 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/wordpred2B.txt"
with open(file2write3, 'w') as f2w3:
	f2w3.write(predword2)

'''
#Train the model
teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, \
			criterion, max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	#print(input_length)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = \
				decoder(decoder_input, decoder_hidden, encoder_outputs)
			#print(decoder_output.type(), decoder_output.size())
			#print(target_tensor[di].type(), target_tensor[di].size())
			loss += criterion(decoder_output, target_tensor[di])
			decoder_input = target_tensor[di]  # Teacher forcing

	else:
		# Without teacher forcing: use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = \
				decoder(decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # detach from history as input

			loss += criterion(decoder_output, target_tensor[di])
			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

#helper function to print time elapsed and estimared time remaining
import time
import math

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

#Train iterativly
def trainIters(encoder, decoder, n_iters, Pairs, print_every=1000, plot_every=100, learning_rate=0.01):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair(Pairs[i]) \
					  for i in range(n_iters)]
	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),\
										 iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
	#showPlot(plot_losses)

#Plot the results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
	plt.figure()
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	loc = ticker.MultipleLocator(base=0.2)
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)


#Initialize and start training
hidden_size = 256
#posPair model
encoder1 = EncoderRNN(n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, dropout_p=0.1).to(device)
#negPair model
encoder2 = EncoderRNN(n_words, hidden_size).to(device)
attn_decoder2 = AttnDecoderRNN(hidden_size, n_words, dropout_p=0.1).to(device)

epochs = 25
for epoch in range(epochs):
	trainIters(encoder1, attn_decoder1, lenofpos, posPairs, print_every=100)
	trainIters(encoder2, attn_decoder2, lenofneg, negPairs, print_every=100)
'''