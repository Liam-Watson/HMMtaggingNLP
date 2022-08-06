# Hidden Markov Model for ixiXhosa POS tagging
Repo: https://github.com/Liam-Watson/HMMtaggingNLP
## Explanation of repositiory
### Python files
There are three main files - preprocessing.py train.py test.py
* preprocessing.py converts csv into correct format, removed punc and adds UNK tokens
* train.py forms frequency tables, smoothes data and calculates the transition and emission probabilities
* test.py uses the tables to perform POS tagging on the test dataset and calculate the model accuracy
### Data
* processed train.npy and test.npy files. For development (validation) just use a subset of the testset
* testPredictions.txt containing best predictions on the test set as well as accuracies for each sentence.
* The train and test csv's  
## Instructions
### Enviroment
* This project was developed on Ubuntu 22.04 with an anaconda enviroment,
* Any unix enviroment should surfice.
* Python version 3.10.4
* Python packages needed:
 * numpy
 * pandas
## Execution order
* Download isiXhosa dataset from https://repo.sadilar.org/
* Convert to csv format with # as delimiter and "" used for string indication (I used libre office calc) 
* `python3 preprocessing.py <path to train csv> <path to test csv> <random UNK proportion>`
* `python3 train.py <smoothing type = laplace | add_k | AD | GT>`
* `python3 test.py <smoothing type>`


