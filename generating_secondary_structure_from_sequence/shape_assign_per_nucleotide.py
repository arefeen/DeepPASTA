#!/usr/bin/env python

import sys;
import subprocess;
import pydoc;
import os;
import re;
import random;
import numpy;
import math;
from Bio import SeqIO;
from Bio.SeqRecord import SeqRecord;


def replaceSupportingCharacter(finalShape):
	for i in range(len(finalShape)):
		if finalShape[i] == '<':
			finalShape[i] = 'L'
		elif finalShape[i] == '>':
			finalShape[i] = 'R'
	return finalShape

def updateForIAndM(finalShape, pattern):
	firstIndex = -1
	for i in range(len(finalShape)):
		if finalShape[i] == '(' and finalShape[i+1] != '(':
			firstIndex = (i+1)
		elif (finalShape[i] == '.' or finalShape[i] == '>') and finalShape[i+1] == ')' and firstIndex != -1:
			lastIndex = i
			modified = False
			if len(re.findall(pattern, ''.join(finalShape)[firstIndex:lastIndex])) == 1:
				l = firstIndex
				while (l <= lastIndex):
					if finalShape[l] == '<':
						finalShape[l] = 'L'
					elif finalShape[l] == '>':
						finalShape[l] = 'R'
					elif finalShape[l] == '.':
						finalShape[l] = 'I'
					modified = True
					l = l + 1
			elif len(re.findall(pattern, ''.join(finalShape)[firstIndex:lastIndex])) > 1:
				l = firstIndex
				while (l <= lastIndex):
					if finalShape[l] == '<':
						finalShape[l] = 'L'
					elif finalShape[l] == '>':
						finalShape[l] = 'R'
					elif finalShape[l] == '.':
						finalShape[l] = 'M'
					modified = True
					l = l + 1
			if modified == True:
				l = firstIndex - 1
				m = lastIndex + 1
				while (finalShape[l] == '(' and finalShape[m] == ')'):
					finalShape[l] = '<'
					finalShape[m] = '>'
					l = l - 1
					m = m + 1
			firstIndex = -1
		elif finalShape[i] == '(' and finalShape[i+1] == '(':
			firstIndex = -1
	return finalShape

def assignUnpairedPosition(shape):
	for i in range(len(shape)):
		if shape[i] == '.':
			shape[i] = 'U'
	return shape

def shapeAtEachPosition(abstractShape):
	beforeOrAfterAnyParanthesis = False
	finalShape = ['*'] * len(abstractShape)
	lastParanthesisIndex = 0
	for i in range(len(abstractShape)):
		if abstractShape[i] == ')':
			lastParanthesisIndex = i
		finalShape[i] = abstractShape[i]
	lastbracket = ''
	firstIndex = 0
	lastIndex = 0
	for i in range(len(abstractShape)):
		if beforeOrAfterAnyParanthesis == False and abstractShape[i] == '(':
			beforeOrAfterAnyParanthesis = True
		if beforeOrAfterAnyParanthesis == True and i > lastParanthesisIndex:
			beforeOrAfterAnyParanthesis = False
			
		if abstractShape[i] == "." and beforeOrAfterAnyParanthesis == False:
			finalShape[i] = 'E'
		elif (abstractShape[i] == '(' or abstractShape[i] == ')') and abstractShape[i+1] == '.':
			lastbracket = abstractShape[i]
			firstIndex = (i+1)
		elif abstractShape[i] == '.' and (abstractShape[i+1] == ')' or abstractShape[i+1] == '('):
			lastIndex = i
			if lastbracket == '(' and abstractShape[i+1] == ')':
				l = firstIndex
				while (l <= lastIndex):
					finalShape[l] = 'H'
					l = l+1
				l = firstIndex - 1
				m = lastIndex + 1
				while (abstractShape[l] == '(' and abstractShape[m] == ')'):
					finalShape[l] = '<'
					finalShape[m] = '>'
					l = l - 1
					m = m + 1
	
			lastbracket = ''
	
	finalShape = updateForIAndM(finalShape, '<+\w+>+')
	count = 0
	while ('(' in ''.join(finalShape)) or (')' in ''.join(finalShape)):
		newFinalShape = updateForIAndM(finalShape, '<+\w*>+')
		if newFinalShape == finalShape:
			if count < 30:
				count = count + 1
			else:
				break
		finalShape = newFinalShape
	finalShape = replaceSupportingCharacter(finalShape)	
	if '.' in ''.join(finalShape):
		finalShape = assignUnpairedPosition(finalShape)

	return finalShape

argvi = 1
inputFile = ""
outputFile = ""
NUMBER_OF_STRUCTURE = 3
print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i]=='-i' or sys.argv[i]=='--inputfile':
			inputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output from RNAshapes to assign structure: " + inputFile)
		if sys.argv[i] == '-o' or sys.argv[i] == '--outputfile':
			outputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output file name: " + outputFile)
		if sys.argv[i] == '-c' or sys.argv[i] == '--structurecount':
			NUMBER_OF_STRUCTURE = int(sys.argv[i+1])
			print>>sys.stderr, ("Number of structure: " + str(NUMBER_OF_STRUCTURE))


count = 0
outputList = []
lastShape = ''
if inputFile != "":
	for line in open(inputFile):
		if ">" in line:
			if count > 0 and count < NUMBER_OF_STRUCTURE:
				i = count
				while (i < NUMBER_OF_STRUCTURE):
					outputList.append(lastShape)
					i = i + 1
			outputList.append(line)
			count = 0
		elif "(" in line or ")" in line or "." in line:
			lastShape = ''.join(shapeAtEachPosition(line))
			outputList.append(lastShape)
			count = count + 1

if count > 0 and count < NUMBER_OF_STRUCTURE:
	i = count
	while (i < NUMBER_OF_STRUCTURE):
		outputList.append(lastShape)
		i = i + 1

if outputFile != "":
	fid = open(outputFile, "w")
	for line in outputList:
		fid.write(line)
	fid.close() 
