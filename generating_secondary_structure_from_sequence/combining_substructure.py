#!/usr/bin/env python

import sys;
import subprocess;
import pydoc;
import os;


argvi = 1
inputFile = ""
outputFile = ""
print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i]=='-i' or sys.argv[i]=='--inputfile':
			inputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output from RNAshapes: " + inputFile)
		if sys.argv[i] == '-o' or sys.argv[i] == '--outputfilename':
			outputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output filename: " + outputFile)



firstSubStructure = []
secondSubStructure = []
outputList = []
current = 'none'
if inputFile != "":
	for line in open(inputFile):
		if ">" in line:
			if current == 'second':
				for first in firstSubStructure:
					for second in secondSubStructure:
						outputList.append(first+second)
				current = 'none'
			outputList.append(line)
			firstSubStructure = []
			secondSubStructure = []
		elif ('101' in line) and ('200' in line):
			current = 'second'
		elif ('1' in line) and ('100' in line):
			current = 'first'
		elif ("(" in line or ")" in line or "." in line):
			if current == 'first':
				firstSubStructure.append(line[:-1])
			else:
				secondSubStructure.append(line)

if current == 'second':
	for first in firstSubStructure:
		for second in secondSubStructure:
			outputList.append(first+second)
	current = 'none'

if outputFile != "":
	fid = open(outputFile, "w")
	for line in outputList:
		fid.write(line)
	fid.close()
