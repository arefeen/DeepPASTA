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

	activa_conv = PReLU(alpha_initializer = 'zero', weights = None)(seqCov)
	
	seqPool = MaxPooling1D(pool_size = 10, strides = 5)(activa_conv)
	seqPoolDout = Dropout(rate = 0.2)(seqPool)
	
	seqBiLstm = Bidirectional(LSTM(units = 128, return_sequences = True))(seqPoolDout)
	seqBiLstmDout = Dropout(rate = 0.5)(seqBiLstm)
	seqFlat = Flatten()(seqBiLstmDout)
	
	seqDen1 = Dense(256, kernel_initializer='glorot_uniform', activation = 'relu')(seqFlat)
	seqDout1 = Dropout(rate = 0.5)(seqDen1)

	return seqDout1

def combinedModel(seqInput, structInput):
	layer_list = []
	
	layer_list.append(sequenceModel(seqInput))

	layer_list.append(structureModel(structInput))
	merged_layer = concatenate(layer_list)

	comDen1 = Dense(128, kernel_initializer='glorot_uniform', activation = 'relu')(merged_layer)
	comDout1 = Dropout(rate = 0.5)(comDen1)


	comDen2 = Dense(64, kernel_initializer='glorot_uniform', activation = 'relu')(comDout1)
	comDout2 = Dropout(rate = 0.5)(comDen2)

	comDen4 = Dense(1, kernel_initializer='glorot_uniform')(comDout2)

	return comDen4


def structureModel(structInput):
	structCov = Conv1D(filters = 16,
			   kernel_size = 12,
			   padding = 'valid',
			   activation = 'relu',
			   strides = 1)(structInput)

	structPool = AveragePooling1D(pool_size = 20, strides = 10)(structCov)
	structPoolDout = Dropout(rate=0.2)(structPool)
	
	structBiLstm = Bidirectional(LSTM(units = 8, return_sequences = True))(structPoolDout)
	structBiLstmDout = Dropout(rate = 0.5)(structBiLstm)
	structFlat = Flatten()(structBiLstmDout)

	structDen1 = Dense(2, kernel_initializer='glorot_uniform')(structFlat)
	structActivaDen1 = PReLU(alpha_initializer = 'zero', weights= None)(structDen1)
	structDout1 = Dropout(rate=0.9)(structActivaDen1)

	return structDout1


### main function ####
argvi = 1
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
seqInput1 = Input(shape = (200, 5))
structInput1 = Input(shape = (200, 7))
comModel1 = combinedModel(seqInput1, structInput1)
training_net.append(comModel1)
# deep learning sub-model for structure
seqInput2 = Input(shape = (200, 5))
structInput2 = Input(shape = (200, 7))
comModel2 = combinedModel(seqInput2, structInput2)
training_net.append(comModel2)
merged_model = concatenate(training_net)

den1_1 = Dense(2, activation = 'softmax')(merged_model)

model = Model(inputs = [seqInput1,structInput1, seqInput2,structInput2], outputs = den1_1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights("DeepPASTA_relatively_maqc_brain2_learned.hdf5")

testingSequenceList1 = []
testingSequenceList2 = []
testingStructList1 = []
testingStructList2 = []
testingReadList = []
geneNameList = []
endPointList = []

if testingFile != "":
	if typeOfOperation == 'DEFAULT':
		for line in open(testingFile):
			field = line.strip().split()
			geneNameList.append(field[0])
			read1 = (1.0 + long(field[5]) * 1.0) / (2.0 + long(field[5]) + long(field[6]))
			read2 = (1.0 + long(field[6]) * 1.0) / (2.0 + long(field[5]) + long(field[6]))
			if long(field[1]) < long(field[2]):
				testingSequenceList1.append(field[3])
				testingSequenceList2.append(field[4])
				testingStructList1.append(field[7])
				testingStructList2.append(field[8])
				testingReadList.append([read1, read2])
				endPointList.append([field[1], field[2]])
			else:
				testingSequenceList1.append(field[4])
				testingSequenceList2.append(field[3])
				testingStructList1.append(field[8])
				testingStructList2.append(field[7])
				testingReadList.append([read2, read1])
				endPointList.append([field[2], field[1]])
	else:
		for line in open(testingFile):
			field = line.strip().split()
			geneNameList.append(field[0])
			if long(field[1]) < long(field[2]):
				testingSequenceList1.append(field[3])
				testingSequenceList2.append(field[4])
				testingStructList1.append(field[5])
				testingStructList2.append(field[6])
				endPointList.append([field[1], field[2]])
			else:
				testingSequenceList1.append(field[4])
				testingSequenceList2.append(field[3])
				testingStructList1.append(field[6])
				testingStructList2.append(field[5])
				endPointList.append([field[2], field[1]])

print>>sys.stderr, 'Testing size: ' + str(len(testingSequenceList1))

encodedTesting1 = seqEncoder(testingSequenceList1)
encodedTesting2 = seqEncoder(testingSequenceList2)

encodedTestingStructure1 = structEncoder(testingStructList1)
encodedTestingStructure2 = structEncoder(testingStructList2)

testingData = []
testingLabel = []
testingData.append(encodedTesting1)
testingData.append(encodedTestingStructure1)
testingData.append(encodedTesting2)
testingData.append(encodedTestingStructure2)

preds = model.predict(testingData, batch_size = 100, verbose = 0)

if typeOfOperation == 'DEFAULT':
	groundTruthList = []
	predictionList = []
	for i in range(len(preds)):
		if testingReadList[i][0] > testingReadList[i][1]:
			groundTruthList.append(1.0)
		else:
			groundTruthList.append(0.0)
		predictionList.append(preds[i][0])
	print 'AUC: ' + str(roc_auc_score(groundTruthList, predictionList))
	print 'AUPRC: ' + str(average_precision_score(groundTruthList, predictionList))
else:
	fid = open(outputFile, 'w')
	for i in range(len(preds)):
		fid.write(geneNameList[i] + '\t' + ','.join(str(e) for e in endPointList[i]) + '\t' + ','.join(str(round(read, 4)) for read in preds[i]) + '\n')
	fid.close()

