# DeepPASTA: Deep neural network based polyadenylation site analysis

According to the central dogma of molecular biology, the genomic sequence of an eukaryotic gene is transformed into the corresponding 
protein by the transcription, post-transcriptional and translation processes. One of the important steps of the post-transcriptional 
process is the polyadenylation process. This process consists of two steps: cleavage near the 3' end of a pre-mRNA and addition of a
polyA tail at the cleavage site or polyA site. DeepPASTA is a tool for predicting such polyA sites from sequence and RNA secondary 
structure data. The tool can also predict tissue-specific polyA sites from sequence and RNA secondary structure data. When two polyA
sites of a gene (of a particular tissue) are given the tool can predict relatively dominant polyA site. Moreover, when a polyA site
of a gene (of a particular tissue) is given it can predict whether that site is one of the most (absolutely) dominant polyA sites 
of the gene or not.


### Requirements
1. The tool runs on linux machine.
2. [Anaconda2-4.4.1](https://docs.anaconda.com/anaconda/install/linux/). 
3. After installing Anaconda2, please use DeepPASTA_env.yml file to create DeepPASTA suitable environment. 
   Please run the following command to activate the suitable environment:

		conda env create -f DeepPASTA_env.yml
		source activate DeepPASTA_env
4. Please [**download**](https://www.cs.ucr.edu/~aaref001/DeepPASTA_site.html) the trained parameters and put them in respective folders. 

## PolyA site prediction
In order to predict polyA sites, please use DeepPASTA_polyA_site_prediction_testing.py of polyA_site_prediction folder. 
Sample input files are given in the sample_input directory. Commands to run the polyA site prediction tool:

USAGE

	cd polyA_site_prediction
	python DeepPASTA_polyA_site_prediction_testing.py {OPTIONS}	


OPTIONS

	-testSeq <input_sequence_file>	A FASTA file that contains human genomic sequences of length 200 nts. 

	-testSS <input_RNA_secondary_structure_file>	An input file that contains the RNA secondary structures of the input sequences.
					The tool expects three most energy efficient RNA secondary structures for each input sequence.
					These RNA secondary structures are generated using [RNAshapes](https://academic.oup.com/bioinformatics/article/22/4/500/184565).

	-o <output_file_name>		Output file name is given using this option. If this option is not used then the tool outputs
					AUC and AUPRC values of the prediction. In order to calculate the AUC and AUPRC values the tool 
					needs ground truth data. The ground truth data is added at the end of the title of each
					sequence. E.g. for a positive sequence example, the title is >chr15_100354095_positive_1; on
					the other hand, the title of a negative sequence example is >chr15_100565120_positive_0. 

EXAMPLE

	python DeepPASTA_polyA_site_prediction_testing.py -testSeq sample_sequence_input.hg19.fa -testSS sample_secondary_structure_input.txt  


### Input and output file of the polyA site prediction model
The model takes two files as input: sequence and RNA secondary structure files. The sequence file is a FASTA file that contains two lines per example.
The first line is the title of the sequence and the second line is the 200 nts sequence. The RNA secondary structure has four lines per example.
The first line is the title and the next three lines for three RNA secondary structures. The model outputs AUC and AUPRC values when -o option
is not used. In order to get the AUC and AUPRC values, the user must give the ground truth values using the title. E.g. title_ground_truth_value; 
for a positive sequence example, the title is >chr15_100354095_positive_1; on the other hand, the title of a negative sequence example is >chr15_100565120_positive_0. 
If the user uses -o option, the model will output the predicted likelihood values in an output file. 

	
## Tissue-specific polyA site prediction
In order to predict tissue-specific polyA sites, please use DeepPASTA_tissue_specific_polyA_site_prediction_testing.py of tissue_specific_polyA_site_prediction folder. 
Sample input files are given in the sample_input directory. Commands to run the tissue-specific polyA site prediction tool:

USAGE

	cd tissue_specific_polyA_site_prediction
	python DeepPASTA_tissue_specific_polyA_site_prediction_testing.py {OPTIONS}

OPTIONS

	-test <input_sequence_file>	A FASTA file that contains human genomic sequences of length 200 nts.
	
	-tests <input_RNA_secondary_structure_file>	An input file that contains the RNA secondary structures of the input sequences. 
                                                The tool expects three most energy efficient RNA secondary structures for each input sequence.
						These RNA secondary structures are generated using [RNAshapes](https://academic.oup.com/bioinformatics/article/22/4/500/184565).

	-testl <ground_truth_label_file>	An input file for the ground truth labels. The ground truth labels help the tool to calculate AUC and AUPRC values.

	-o <output_file_name>	Output file name is given using this option. This option prints the result in an output file.

EXAMPLE

	python DeepPASTA_tissue_specific_polyA_site_prediction_testing.py -test sample_sequence_input.hg19.fa -tests sample_secondary_structure_input.txt -testl sample_tissue_specific_label.txt

### Input and output of tissue-specific polyA site prediction model
The model takes two files: sequence and RNA secondary structure files. The sequence file is a FASTA file that contains two lines per example.
The first line is the title of the sequence and the second line is the 200 nts sequence. The RNA secondary structure has four lines per example.
The first line is the title and the next three lines for three RNA secondary structures. The model outputs AUC and AUPRC values when -testl option
is used. Using -testl option the user have to give the ground truth data. For each example, the ground truth data has two lines: these two lines are 
title and read counts (of nine tissues separated by comma), respectively. If -o option is used the model outputs tissue-specific polyA site prediction 
in a file. For each input example the output file has a line containing title and nine likelihood values (separated by comma) for nine tissues.


## Tissue-specific relatively dominant polyA site prediction
The tool can also predict tissue-specific relatively dominant polyA sites. The files to run the tissue-specific relatively dominant polyA
site prediction model are in tissue_specific_relatively_dominant folder. For an example, if an user wants to run the liver tissue relatively
dominant polyA site prediction model, he/she have to follow the following commands:

USAGE
	
	cd tissue_specific_relatively_dominant/tissue_set_one/liver
	python DeepPASTA_relatively_dominant_liver_testing.py {OPTIONS}

OPTIONS
	
	-test <input_file>	An input file that contains gene name, two polyA sites, two sequences around the polyA sites, the sequences 
				corresponding RNA secondary structures. If the user wants to caculate the AUC and AUPRC values of the prediction,
				he/she also have to provide the polyA sites corresponding read counts in this file.

	-o <output_file_name>	This option prints the result in an output file.

EXAMPLE

	python DeepPASTA_relatively_dominant_liver_testing.py -test sample_relatively_dominant_input_liver.txt

### Input and output of tissue-specific relatively dominant polyA site prediction model
If the user wants to calculate the AUC and AUPRC values of the prediction, he/she must input a file that contains gene name, first polyA site location, second
polyA site location, sequence (200 nts) around the first polyA site, sequence (200 nts) around the second polyA site, read count of the first polyA site, 
read count of the second polyA site, RNA secondary structure of the first sequence, and RNA secondary structure of the second sequence. If the user wants 
to output the prediction result to a file using -o option, the input file must contains all the above information except the read counts. The output file contains 
gene name, the two polyA site locations and the probabilities of relative dominance of these two polyA sites. Sample input files of the tissue-specific relatively 
dominant polyA site prediction model are given in the sample_input folder. 


## Tissue-specific absolutely dominant polyA site prediction
The tool can predict tissue-specific absolutely dominant polyA site when sequence (and RNA secondary structure) around a polyA site of a gene 
is given as input. The files to run the tissue-specific absolutely dominant polyA site prediction model are in tissue_specific_absolutely_dominant
folder. For an example, if an user wants to run the liver tissue absolutely dominant polyA site prediction model, he/she have to follow the following
commands:

USAGE

	cd tissue_specific_absolutely_dominant/liver
	python DeepPASTA_absolutely_dominant_liver_testing.py {OPTIONS}

OPTOINS

	-test <input_file>	An input file that contains gene name, tissue name, polyA site location, sequence around the polyA site, RNA secondary structure 
				of the sequence. If the user wants to calculte the AUC and AUPRC values of the prediction, he/she have to provide the
				label of the polyA site in this file.

	-o <output_file_name>	This option prints the result in an output file.

EXAMPLE

	python DeepPASTA_absolutely_dominant_liver_testing.py -test sample_absolutely_dominant_input_liver.txt

### Input and output of tissue-specific absolutely dominant polyA site prediction model
If the user wants to calculate the AUC and AUPRC values of the prediction, he/she must input a file that contains gene name, tissue name, polyA site location, sequence
(200 nts) around the polyA site, RNA secondary structure of the sequence, and absolutely dominant label. If the user wants to output the prediction result to a file using
-o option, the input file must contains all the above information except the label. The output file contains gene name, polyA site location and likelihood value of the 
prediction. Sample input files of the tissue-specific absolutely dominant polyA site prediction model are given in the sample_input folder.



Note: If you have any question or suggestion please feel free to email: aaref001@ucr.edu or ashraful.arefeen@csebuet.org
