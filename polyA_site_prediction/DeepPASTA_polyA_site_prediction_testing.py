#!/usr/bin/env python

import sys
import numpy as np
import math
import random

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Bidirectional, Input, concatenate, add
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import PReLU
from sklearn.metrics import average_precision_score, roc_auc_score


np.random.seed(1337)
random.seed(1337)

def oneHotEncodingForSeq(rawSeqList):
	if len(rawSeqList) != 0:
		encodedSeq = np.zeros((len(rawSeqList), len(rawSeqList[0]), 5))
		for i in range(len(rawSeqList)):
			sequence = rawSeqList[i]
			j = 0
		    	for s in sequence:
				if s == 'A' or s == 'a':
					encodedSeq[i][j] = [1,0,0,0,0]
				elif s == 'T' or s == 't':
					encodedSeq[i][j] = [0,1,0,0,0]
				elif s == 'C' or s == 'c':
					encodedSeq[i][j] = [0,0,1,0,0]
				elif s == 'G' or s == 'g':
					encodedSeq[i][j] = [0,0,0,1,0]
				elif s == 'N' or s == 'n':
					encodedSeq[i][j] = [0,0,0,0,1]
				else:
					print>>sys.stderr, 'ERROR: Unwanted nucleotide: ' + s
				j = j + 1
		return encodedSeq
	else:
		return 0

def oneHotEncodingForSS(rawStructureList):
	if len(rawStructureList) != 0:
		encodedStructure = np.zeros((len(rawStructureList), len(rawStructureList[0]), 7))
		for i in range(len(rawStructureList)):
			structure = rawStructureList[i]
			j = 0
		    	for s in structure:
				if s == 'U':
					encodedStructure[i][j] = [1,0,0,0,0,0,0]
				elif s == 'E':
					encodedStructure[i][j] = [0,1,0,0,0,0,0]
				elif s == 'L':
					encodedStructure[i][j] = [0,0,1,0,0,0,0]
				elif s == 'R':
					encodedStructure[i][j] = [0,0,0,1,0,0,0]
				elif s == 'H':
					encodedStructure[i][j] = [0,0,0,0,1,0,0]
				elif s == 'M':
					encodedStructure[i][j] = [0,0,0,0,0,1,0]
				elif s == 'I':
					encodedStructure[i][j] = [0,0,0,0,0,0,1]
				else:
					print>>sys.stderr, 'Warning: Unwanted character ' + s
				j = j + 1
		return encodedStructure
	else:
		return 0

def matchingLabelBetweenSeqAndStructure(seqLabelList, structureLabelList, kindOfData):
	print>>sys.stderr, 'Checking label similarity between sequence and structure of ' + kindOfData + ' data'
	for index in range(len(seqLabelList)):
		if seqLabelList[index] != structureLabelList[index]:
			print>>sys.stderr, 'ERROR: label mismatch between sequence and structure'

def sequenceModel(seqInput):
	seqCov = Conv1D(filters=512,
		kernel_size=8,
		padding = "valid",
		input_shape =(200, 5),
		activation="relu",
		strides=1)(seqInput) 
	
	seqPool = MaxPooling1D(pool_size = 3, strides = 3)(seqCov)
	seqDout1 = Dropout(rate = 0.7)(seqPool)
	seqBiLstm = Bidirectional(LSTM(units = 128, return_sequences = True))(seqDout1)
	seqDout2 = Dropout(rate = 0.7)(seqBiLstm)
	seqFlat = Flatten()(seqDout2)
	seqDen2 = Dense(256, kernel_initializer='glorot_uniform', activation = 'relu')(seqFlat)
	seqDout4 = Dropout(rate = 0.7)(seqDen2)

	return seqDout4

def structureSubModel(ssInput):
	ssConv = Conv1D(filters=256,
	               	kernel_size=12,
			padding = "valid",
			activation="relu",
			strides=1)(ssInput)
	ssPool = AveragePooling1D(pool_size = 5, strides = 5)(ssConv)
	ssDout1 = Dropout(rate=0.7)(ssPool)
	seqBiLstm = Bidirectional(LSTM(units = 128, return_sequences = True))(ssDout1)
	seqDout2 = Dropout(rate = 0.7)(seqBiLstm)
	ssFlat = Flatten()(seqDout2)
	ssDen1 = Dense(256, kernel_initializer='glorot_uniform', activation = 'relu')(ssFlat)
	ssDout2 = Dropout(rate=0.7)(ssDen1)

	return ssDout2

### main function ####
argvi = 1
testingSeqFile = ''
testingStructureFile = ''
typeOfOperation = 'DEFAULT'
outputFile = ''
structureSeqLength = 200

print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i] == '-testSeq' or sys.argv[i] == '--testingSeqFASTAFile':
			testingSeqFile = sys.argv[i+1]
			print>>sys.stderr, ("Genome sequence file (testing): " + testingSeqFile)
		if sys.argv[i] == '-testSS' or sys.argv[i] == '--testingStructureFile':
			testingStructureFile = sys.argv[i+1]
			print>>sys.stderr, ("Secondary structure file for testing (related to sequence file): " + testingStructureFile)
		if sys.argv[i] == '-o' or sys.argv[i] == '--outputFile':
			outputFile = sys.argv[i+1]
			print>>sys.stderr, ("Predicted result output file name: " + outputFile)

if outputFile != '':
	typeOfOperation = 'PRINT'

# Building deep learning model
training_net = []
# deep learning sub-model for sequence
seqInput = Input(shape = (200, 5))
seqModel = sequenceModel(seqInput)
training_net.append(seqModel)

# deep learning sub-model for structure
ss_training_net = []
ssInput1 = Input(shape = (structureSeqLength, 7))
ssInput2 = Input(shape = (structureSeqLength, 7))
ssInput3 = Input(shape = (structureSeqLength, 7))

ss_training_net.append(structureSubModel(ssInput1))
ss_training_net.append(structureSubModel(ssInput2))
ss_training_net.append(structureSubModel(ssInput3))

ss_merged_model = add(ss_training_net)
ss_den1 = Dense(256, kernel_initializer = 'glorot_uniform', activation = 'relu')(ss_merged_model)
ss_dout1 = Dropout(rate = 0.7)(ss_den1)
training_net.append(ss_dout1)
merged_model = concatenate(training_net)

den1 = Dense(256, kernel_initializer = 'glorot_uniform', activation = 'relu')(merged_model)
dout1 = Dropout(rate = 0.7)(den1)

den2 = Dense(128, kernel_initializer = 'glorot_uniform', activation = 'relu')(dout1)
dout2 = Dropout(rate = 0.7)(den2)
den3 = Dense(64, activation = 'relu')(dout2)
den4 = Dense(1, activation = 'sigmoid')(den3)
model = Model(inputs = [seqInput, ssInput1, ssInput2, ssInput3], outputs = den4)

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.load_weights('DeepPASTA_polyA_site_learned.hdf5')

testingSequenceList = []
testingLabelList = []
testingTitleList = []
if testingSeqFile != '':
	for line in open(testingSeqFile):
		info = line[0:(len(line)-1)]
		if '>' in info:
			if typeOfOperation == 'DEFAULT':
				testingLabelList.append(int(info[-1:]))
			testingTitleList.append(info)
		else:
			testingSequenceList.append(info)
else:
	print>>sys.stderr, 'ERROR: No testing sequence file is given'


testingStructureList1 = []
testingStructureList2 = []
testingStructureList3 = []
testingSSLabelList = []

listNumber = 0
if testingStructureFile != "":
	for line in open(testingStructureFile):
		field = line.strip().split()
		if '>' in field[0]:
			testingSSLabelList.append(int(field[0][-1:]))
			listNumber = 1
		else:
			if listNumber == 1:
				testingStructureList1.append(field[0])
				listNumber = 2
			elif listNumber == 2:
				testingStructureList2.append(field[0])
				listNumber = 3
			elif listNumber == 3:
				testingStructureList3.append(field[0])
				listNumber = 0
else:
	print>>sys.stderr, 'ERROR: validation file is not given'

encodedTestingSeq = oneHotEncodingForSeq(testingSequenceList)
encodedTestingStructure1 = oneHotEncodingForSS(testingStructureList1)
encodedTestingStructure2 = oneHotEncodingForSS(testingStructureList2)
encodedTestingStructure3 = oneHotEncodingForSS(testingStructureList3)

testingData = []
testingData.append(encodedTestingSeq)
testingData.append(encodedTestingStructure1)
testingData.append(encodedTestingStructure2)
testingData.append(encodedTestingStructure3)

testresult1 = model.predict(testingData, batch_size = 2042, verbose = 0)

if typeOfOperation == 'DEFAULT':
	print 'AUC: ' + str(roc_auc_score(testingLabelList, testresult1))
	print 'AUPRC: ' + str(average_precision_score(testingLabelList, testresult1))
else:
	fid = open(outputFile, 'w')
	for i in range(len(testresult1)):
		fid.write(testingTitleList[i] + '\t' + str(testresult1[i][0]) + '\n')
	fid.close()

