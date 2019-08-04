#!/usr/bin/env python

import sys;
import subprocess;
import pydoc;
import os;


argvi = 1
inputFile = ""
outputFile = ""
numberOfStructures = 5
print>>sys.stderr, "Starting program ..."
if len(sys.argv) < 2:
	print(pydoc.render_doc(sys.modules[__name__]));
	sys.exit();

for i in range(len(sys.argv)):
	if i < len(sys.argv)-1:
		if sys.argv[i]=='-i' or sys.argv[i]=='--inputfile':
			inputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output from RNAshapes: " + inputFile)
		if sys.argv[i] == '-n' or sys.argv[i] == '--numberofstructure':
			numberOfStructures = int(sys.argv[i+1])
			print>>sys.stderr, ("Number of suboptimal structures considered: " + str(numberOfStructures))
		if sys.argv[i] == '-o' or sys.argv[i] == '--outputfilename':
			outputFile = sys.argv[i+1]
			print>>sys.stderr, ("Output filename: " + outputFile)



count = 0
outputList = []
if inputFile != "":
	for line in open(inputFile):
		if ">" in line:
			outputList.append(line)
			count = 0
		elif ("(" in line or ")" in line or "." in line) and count < numberOfStructures:
			outputList.append(line)
			count = count + 1

if outputFile != "":
	fid = open(outputFile, "w")
	for line in outputList:
		fid.write(line)
	fid.close()
