#!/usr/bin/env python

import sys
import numpy as np
import math
import random

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Bidirectional, Input, concatenate
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import PReLU
from sklearn.metrics import average_precision_score, roc_auc_score


np.random.seed(1337)
random.seed(1337)

def seqEncoder(rawSeqList):
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

def structEncoder(rawStructureList):
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


def sequenceModel(seqInput):
	seqCov = Conv1D(filters=256,
		kernel_size=8,
		padding = "valid",
		input_shape =(200, 5),
		strides=1)(seqInput) 

	activa1 = PReLU(alpha_initializer = 'zero', weights = None)(seqCov)
	seqPool = MaxPooling1D(pool_size = 3, strides = 3)(activa1)
	seqDout1 = Dropout(rate = 0.2)(seqPool)
	seqBiLstm = Bidirectional(LSTM(units = 128, return_sequences = True))(seqDout1)
	seqDout2 = Dropout(rate = 0.5)(seqBiLstm)
	seqFlat = Flatten()(seqDout2)
	seqDen3 = Dense(256, kernel_initializer='glorot_uniform')(seqFlat)
	activa2 = PReLU(alpha_initializer = 'zero', weights= None)(seqDen3)
	seqDout3 = Dropout(rate = 0.5)(activa2)

	return seqDout3

def structureModel(InputShape):
	conv1 = Conv1D(filters=256,
	               	kernel_size= 12,
			padding = 'valid',
			activation='relu',
			strides=1)(InputShape)
	
	mapool = AveragePooling1D(pool_size = 5, strides = 5)(conv1)
	dout1 = Dropout(rate=0.2)(mapool)
	
	structBiLstm = Bidirectional(LSTM(units = 128, return_sequences = True))(dout1)
	seqDout2 = Dropout(rate = 0.4)(structBiLstm)
	flat = Flatten()(seqDout2)
	den1 = Dense(256, kernel_initializer='glorot_uniform')(flat)
	activa2 = PReLU(alpha_initializer = 'zero', weights= None)(den1)
	dout2 = Dropout(rate=0.5)(activa2)

	return dout2


### main function ####
argvi = 1
trainingFile = ''
validationFile = ''
testingFile = ''
typeOfOperation = 'DEFAULT'
outputFile = ''
print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i] == '-test' or sys.argv[i] == '--testingFile':
			testingFile = sys.argv[i+1]
			print>>sys.stderr, ("Testing file: " + testingFile)
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
structInput = Input(shape = (200, 7))
structModel = structureModel(structInput)
training_net.append(structModel)

merged_model = concatenate(training_net)

den1 = Dense(256, kernel_initializer = 'glorot_uniform')(merged_model)
dout1 = Dropout(rate = 0.5)(den1)
den2 = Dense(64, kernel_initializer = 'glorot_uniform', activation = 'relu')(dout1)
dout2 = Dropout(rate = 0.5)(den2)
den3 = Dense(16, kernel_initializer = 'glorot_uniform', activation = 'relu')(dout2)
dout3 = Dropout(rate = 0.5)(den3)

den1_1 = Dense(1, activation = 'sigmoid')(dout3)

model = Model(inputs = [seqInput, structInput], outputs = den1_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('DeepPASTA_absolutely_liver_learned.hdf5')


testingSequenceList = []
testingStructList = []
testingLabelList = []
if typeOfOperation == 'PRINT':
	geneList = []
	endPointList = []

if testingFile != '':
	for line in open(testingFile):
		field = line.strip().split()
		testingSequenceList.append(field[3])
		testingStructList.append(field[4])
		if typeOfOperation == 'PRINT':
			geneList.append(field[0])
			endPointList.append(field[2])
		else:
			testingLabelList.append(long(field[5]))

print>>sys.stderr, 'Testing data size: ' + str(len(testingSequenceList))

encodedTesting = seqEncoder(testingSequenceList)
encodedTestingStruct = structEncoder(testingStructList)

testingData = []
testingData.append(encodedTesting)
testingData.append(encodedTestingStruct)

testresult1 = model.predict(testingData, batch_size = 200, verbose = 0)

if typeOfOperation == 'DEFAULT':
	print 'AUC: ' + str(roc_auc_score(testingLabelList, testresult1))
	print 'AUPRC: ' + str(average_precision_score(testingLabelList, testresult1))
else:
	fid = open(outputFile, 'w')
	for i in range(len(testresult1)):
		fid.write(geneList[i] + '\t' + endPointList[i] + '\t' + str(testresult1[i][0]) + '\n')
	fid.close()

