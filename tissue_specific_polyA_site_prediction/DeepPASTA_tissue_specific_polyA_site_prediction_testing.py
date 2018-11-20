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

def convertStringToList(numberStringList):
	mList = [float(e) if e.isdigit() else e for e in numberStringList.split(',')]
	maximum = -1.0
	for i in range(len(mList)):
		if mList[i] >= 1.0:
			mList[i] = 1.0
		else:
			mList[i] = 0.0
	return mList

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
testingFASTAFile = ''
testingLabelFile = ''
testingStructFile = ''
typeOfOperation = 'DEFAULT'
outputFile = ''
print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i] == '-test' or sys.argv[i] == '--testingFASTAFile':
			testingFASTAFile = sys.argv[i+1]
			print>>sys.stderr, ("Genome sequence FASTA file (testing): " + testingFASTAFile)
		if sys.argv[i] == '-testl' or sys.argv[i] == '--testinglabelFile':
			testingLabelFile = sys.argv[i+1]
			print>>sys.stderr, ("Data label file (testing): " + testingLabelFile)
		if sys.argv[i] == '-tests' or sys.argv[i] == '--testingstructFile':
			testingStructFile = sys.argv[i+1]
			print>>sys.stderr, ("Structure file (testing): " + testingStructFile)
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
struct_net = []
structInput1 = Input(shape = (200, 7))
structModel1 = structureModel(structInput1)
struct_net.append(structModel1)

structInput2 = Input(shape = (200, 7))
structModel2 = structureModel(structInput2)
struct_net.append(structModel2)

structInput3 = Input(shape = (200, 7))
structModel3 = structureModel(structInput3)
struct_net.append(structModel3)

merged_structure_model = concatenate(struct_net)
den_struct = Dense(256, kernel_initializer = 'glorot_uniform', activation = 'relu')(merged_structure_model)
dout_struct = Dropout(rate = 0.5)(den_struct)
training_net.append(dout_struct)

merged_model = concatenate(training_net)

den1 = Dense(64, kernel_initializer = 'glorot_uniform')(merged_model)
dout1 = Dropout(rate = 0.5)(den1)
den2 = Dense(32, kernel_initializer = 'glorot_uniform', activation = 'relu')(dout1)
dout2 = Dropout(rate = 0.5)(den2)
den1_1 = Dense(9, activation = 'sigmoid')(dout2)

model = Model(inputs = [seqInput, structInput1, structInput2, structInput3], outputs = den1_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('DeepPASTA_tissue_specific_polyA_learned.hdf5')

testingSequenceList = []
testingTitleList = []
if testingFASTAFile != "":
	for line in open(testingFASTAFile):
		info = line[0:(len(line)-1)]
		if '>' in info:
			testingTitleList.append(info)
		else:
			testingSequenceList.append(info)


testingLabelList = []
i = 0
if testingLabelFile != '':
	for line in open(testingLabelFile):
		info = line[0:(len(line) - 1)]
		if '>' in info:
			if testingTitleList[i] != info:
				print>>sys.stderr, 'Title mismatch (validation): ' + testingTitleList[i] + ' ' + info
			i = i + 1
		else:
			testingLabelList.append(convertStringToList(info))
print>>sys.stderr, 'Testing data size: ' + str(len(testingLabelList))
i = 0
structureNo = 0
testingStructList1 = []
testingStructList2 = []
testingStructList3 = []
if testingStructFile != '':
	for line in open(testingStructFile):
		info = line[0:(len(line) - 1)]
		if '>' in info:
			if testingTitleList[i] != info:
				print>>sys.stderr, 'Title does not matched between sequence and structure file ' + testingTitleList[i] + ' ' + info
			i = i + 1
			structureNo = 1
		else:
			if structureNo == 1:
				testingStructList1.append(info)
				structureNo = 2
			elif structureNo == 2:
				testingStructList2.append(info)
				structureNo = 3
			elif structureNo == 3:
				testingStructList3.append(info)
				structureNo = 4
			else:
				structureNo = structureNo + 1



encodedTesting = seqEncoder(testingSequenceList)
encodedTestingStruct1 = structEncoder(testingStructList1)
encodedTestingStruct2 = structEncoder(testingStructList2)
encodedTestingStruct3 = structEncoder(testingStructList3)

testingData = []
testingData.append(encodedTesting)
testingData.append(encodedTestingStruct1)
testingData.append(encodedTestingStruct2)
testingData.append(encodedTestingStruct3)

preds = model.predict(testingData, batch_size = 1000, verbose = 0)


TISSUE_NAME = ['brain', 'kidney', 'liver', 'maqc_brain1', 'maqc_brain2', 'maqc_UHR1', 'maqc_UHR2', 'muscle', 'testis']
if typeOfOperation == 'DEFAULT':
	for j in range(len(preds[0])):
		print 'Name of the tissue: ' + TISSUE_NAME[j]
		groundTruthList = []
		predictedList = []
		for i in range(len(preds)):
			groundTruthList.append(testingLabelList[i][j])
			predictedList.append(preds[i][j])
		print 'AUC: ' + str(roc_auc_score(groundTruthList, predictedList))
		print 'AUPRC: ' + str(average_precision_score(groundTruthList, predictedList))

else:
	fid = open(outputFile, 'w')
	for i in range(len(preds)):
		fid.write(testingTitleList[i] + '\t' + ','.join(str(round(e,4)) for e in preds[i]) + '\n')
	fid.close()


