import numpy as np 
import sys 
import random

path = sys.argv[1] # path to the train csv file
pathTest = sys.argv[2] # path to the test csv file
randomProp = float(sys.argv[3]) # the proportion of the data to be used for UNKs
# Read in the data
def process(path, testBool):
    with open(path, 'r') as f:
        lines = f.readlines()
        # Remove the first line
        lines.pop(0)

    newLines = [('^','START')] # add the start symbol as initial step
    sentences = [] # list of sentences which we append lists of tuples to
    for l in lines:
        word = l.split('#')[0] # get the word
        tag = l.split('#')[1].strip() # get the tag without the newline
        if tag != 'PUNCT' and tag != '': # remove punctuation and sentence seperators
            if testBool:
                newLines.append((word,tag)) # append the word and tag to the list
            else:
                if random.random() <= randomProp: # randomly sample 5% of the data for UNK tokens (this random sampling does add some non-determinism to the code)
                    newLines.append(("UNK", tag)) # add the UNK token
                else:
                    newLines.append((word,tag)) # add the word and tag
        elif tag == '':
            newLines.append(('$','END')) # add the end symbol
            sentences.append(newLines) # add the sentence to the list of sentences
            newLines = [('^','START')] # reset the newLines list
    return sentences

sentencesTrain = np.array(process(path, False), dtype=object)
sentencesTest = np.array(process(pathTest, True), dtype=object) # convert the list of sentences to a numpy array
np.save('dataTemp/train.npy', sentencesTrain) # save the numpy array to a file
np.save('dataTemp/test.npy', sentencesTest)