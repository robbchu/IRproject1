'''
This xmlparser.py parse the xml file of the train/dev/test data 
into the 1)Whole corpus words file for later use of training word2vec
		 2)List of pairs of data for the use of subtask ,
		 	i.e. in this case of subtaskB, the pair should be(OrgQ/RelQ)
		 	which contain their attributes and the text content
'''
import xml.etree.ElementTree as ET
import pickle
import operator

trainingdata_2016trainpart1 = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part1.xml'
trainingdata_2016trainpart2 = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-train-part2.xml'
trainingdata_2016dev  = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-dev.xml'
trainingdata_2016test = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2016-Task3-CQA-QL-test.xml'
trainingdata_2015train= '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml'
trainingdata_2015dev  = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml'
trainingdata_2015test = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml'
testdata = '/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/test_data/SemEval2017-task3-English-test-input.xml'


def extractCorpusAllFrom2016(filename, subtaskB=1, OrgQCount=0, RelQuestionCount=0, RelCommentCount=0, NotNoneRelQS=0, NotNoneRelQB=0, NotNoneRelC=0):
	tree = ET.parse(filename)
	root = tree.getroot()

	CorpusTextAll=""
	CourpusTextAttrib=[]

	#some parameters used in parsing to record some counting
	prevORGQ_ID = "Q00"
	for OrgQuestion in root.iter('OrgQuestion'):
		textPair = []
		ORGQ_ID = OrgQuestion.attrib['ORGQ_ID']

		if(ORGQ_ID != prevORGQ_ID):
			OrgQCount+=1
			prevORGQ_ID = ORGQ_ID
			OrgQSubject = OrgQuestion.find('OrgQSubject')
			OrgQBody     = OrgQuestion.find('OrgQBody')	
			CorpusTextAll += ((OrgQSubject.text +" "+ OrgQBody.text) +"\n")

		textPair.extend([ORGQ_ID, OrgQSubject.text, OrgQBody.text])
		
		Thread      = OrgQuestion.find('Thread')
		RelQuestion = Thread.find('RelQuestion')

		RelQuestionCount+=1
		RELQ_ID = RelQuestion.attrib['RELQ_ID']
		RELQ_RANKING_ORDER = RelQuestion.attrib['RELQ_RANKING_ORDER']
		RELQ_CATEGORY	   = RelQuestion.attrib['RELQ_CATEGORY']
		RELQ_RELEVANCE2ORGQ= RelQuestion.attrib['RELQ_RELEVANCE2ORGQ']
		RelQSubject = RelQuestion.find('RelQSubject')
		RelQBody	= RelQuestion.find('RelQBody')

		if(RelQBody.text is None):
			#print("find None RelQBody at{}".format(RELQ_ID))
			RelQBody.text = ""


		textPair.extend([RELQ_ID, RELQ_RANKING_ORDER, RELQ_CATEGORY, RelQSubject.text, RelQBody.text, RELQ_RELEVANCE2ORGQ])
		CourpusTextAttrib.append(textPair)
		textPair=[]

		if 'SubtaskA_Skip_Because_Same_As_RelQuestion_ID' not in Thread.attrib:
			if RelQSubject.text is not None:
				NotNoneRelQS+=1
				CorpusTextAll += (RelQSubject.text +" ")
			if RelQBody.text is not None:
				NotNoneRelQB+=1
				CorpusTextAll += (RelQBody.text +"\n")
		
		#subCorpusText += str(RelQSubject.text + RelQBody.text)

		for RelComment in Thread.iter('RelComment'):
			textPair.extend([RELQ_ID, RELQ_CATEGORY, RelQSubject.text, RelQBody.text])
			RelCommentCount+=1
			RELC_ID = RelComment.attrib['RELC_ID']
			RELC_RELEVANCE2ORGQ = RelComment.attrib['RELC_RELEVANCE2ORGQ']
			RELC_RELEVANCE2RELQ = RelComment.attrib['RELC_RELEVANCE2RELQ']
			RelCText = RelComment.find('RelCText')
			if(RelCText.text is None):
				RelCText.text = ""
			if(subtaskB!=1):
				textPair.extend([RELC_ID, RelCText.text, RELC_RELEVANCE2RELQ, RELC_RELEVANCE2ORGQ])
				CourpusTextAttrib.append(textPair)
				textPair=[]
			if 'SubtaskA_Skip_Because_Same_As_RelQuestion_ID' not in Thread.attrib:
				if RelCText.text is not None:
					NotNoneRelC+=1
					CorpusTextAll += (RelCText.text + "\n")
	print("In this file of {},\nThere are OrgQ:{},RelQ:{},RelC:{},\nNotNoneRelQSubject:{}, NotNoneRelQBody:{}, NotNoneRelC:{}"\
		.format(filename, OrgQCount, RelQuestionCount, RelCommentCount, NotNoneRelQS, NotNoneRelQB, NotNoneRelC))
	return (CorpusTextAll, CourpusTextAttrib)

def extractCorpusAllFrom2015(filename, subCorpusText="", RelQuestionCount=0, RelCommentCount=0, NotNoneRelQS=0, NotNoneRelQB=0, NotNoneRelC=0):
	tree = ET.parse(filename)
	root = tree.getroot()

	#some parameters used in parsing to record some counting
	for Thread in root.iter('Thread'):
		THREAD_SEQUENCE = Thread.attrib['THREAD_SEQUENCE']

		RelQuestion = Thread.find('RelQuestion')
		RelQuestionCount+=1
		RELQ_CATEGORY = RelQuestion.attrib['RELQ_CATEGORY']

		RelQSubject = RelQuestion.find('RelQSubject')
		RelQBody	= RelQuestion.find('RelQBody')
		#print(type(RelQSubject.text), type(RelQBody.text))
		#print((RelQSubject.text), (RelQBody.text))

		if RelQSubject.text is not None:
			NotNoneRelQS+=1
			subCorpusText += (RelQSubject.text +" ")
		if RelQBody.text is not None:
			NotNoneRelQB+=1
			subCorpusText += (RelQBody.text +"\n")
		
		#subCorpusText += str(RelQSubject.text + RelQBody.text)

		for RelComment in Thread.iter('RelComment'):
			RelCommentCount+=1
			RELC_ID = RelComment.attrib['RELC_ID']
			RELC_RELEVANCE2RELQ = RelComment.attrib['RELC_RELEVANCE2RELQ']
			RelCText = RelComment.find('RelCText')

			if RelCText.text is not None:
				NotNoneRelC+=1
				subCorpusText += (RelCText.text + "\n")
	print("In this file of {},\nThere are RelQ:{},RelC:{},\nNotNoneRelQSubject:{}, NotNoneRelQBody:{}, NotNoneRelC:{}"\
		.format(filename, RelQuestionCount, RelCommentCount, NotNoneRelQS, NotNoneRelQB, NotNoneRelC))
	return subCorpusText



CorpusText2016trainpart1, CorpusTextAttrib2016trainpart1 =  extractCorpusAllFrom2016(trainingdata_2016trainpart1)
CorpusText2016trainpart2, CorpusTextAttrib2016trainpart2 = extractCorpusAllFrom2016(trainingdata_2016trainpart2)
CorpusText2016dev, CorpusTextAttrib2016dev =extractCorpusAllFrom2016(trainingdata_2016dev)
CorpusText2016test, CorpusTextAttrib2016test =extractCorpusAllFrom2016(trainingdata_2016test)
CorpusText2015train = extractCorpusAllFrom2015(trainingdata_2015train)
CorpusText2015dev = extractCorpusAllFrom2015(trainingdata_2015dev)
CorpusText2015test = extractCorpusAllFrom2015(trainingdata_2015test)

CorpusTextAll=""
CorpusTextAll = CorpusText2016trainpart1+CorpusText2016trainpart2+CorpusText2016dev+CorpusText2016test\
				+CorpusText2015train+CorpusText2015dev+CorpusText2015test

CorpusTextAttrib = []
CorpusTextAttrib.extend(CorpusTextAttrib2016trainpart1)
CorpusTextAttrib.extend(CorpusTextAttrib2016trainpart2)
CorpusTextAttrib.extend(CorpusTextAttrib2016dev)
CorpusTextAttrib.extend(CorpusTextAttrib2016test)

print(len(CorpusTextAttrib), len(CorpusTextAttrib[111]), len(CorpusTextAttrib[0][0]))

testCorpus, testTextAttrib = extractCorpusAllFrom2016(testdata)

#Write the TEXT of all corpus to file output
file2write1 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextAll.txt"
with open(file2write1, 'w') as f2w1:
	f2w1.write(CorpusTextAll)

#Write the LIST object (pair) to file output
file2write2 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB.txt"
with open(file2write2, 'wb') as f2w2:
	pickle.dump(CorpusTextAttrib, f2w2)
#ALSO pre-construct the dictionary of the RelQ category
relQcategory_dict={}
for i, pair in enumerate(CorpusTextAttrib):
	if(pair[5] not in relQcategory_dict):
		relQcategory_dict[pair[5]] = 1
	else:
		relQcategory_dict[pair[5]] += 1
'''
#Build the dictionary of the relQ category
relQcategory_dict = sorted(relQcategory_dict.items(), key=operator.itemgetter(1), reverse=True)
file2write3 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/training_data/corpusfileTextPairB_Catdictionary.txt"
with open(file2write3, 'wb') as f2w3:
	pickle.dump(relQcategory_dict, f2w3)
print(relQcategory_dict[0:5])
'''
'''
#To read the LIST object back
with open(file2write2, 'rb') as f2r1:
	pairList = pickle.load(f2r1)
print(pairList[0:5])
'''
file2write4 = "/home/robert/Documents/2018fall_IRIE/IRIE2018_project_1/test_data/testPair.txt"
with open(file2write4, 'wb') as f2w4:
	pickle.dump(testTextAttrib, f2w4)