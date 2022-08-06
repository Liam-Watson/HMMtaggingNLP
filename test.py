import numpy as np
import pandas as pd
import sys

smoothing = sys.argv[1] # choose the smoothing method

d = pd.read_pickle("dataTemp/emissionTable_" + smoothing + ".pkl") # read in the emission table
d2 = pd.read_pickle("dataTemp/transitionTable_" + smoothing + ".pkl") # read in the transition table
sentences = np.load('dataTemp/test.npy', allow_pickle=True) # read in the test sentences

'''Get the emission probability of a word given a tag'''
dIndex = list(d.index) # get the index of the emission table for efficiency
dCols = list(d.columns) # get the columns of the emission table for efficiency
def emission(word, tag):
    if tag in dIndex:
        if word in dCols:
            return d[word][tag] # return the probability of the tag given the word
        else:
            return 0
            
    else:
        return 0

'''Get the transition probability of the tags'''
d2cols = list(d2.columns) # get the columns of the transition table for efficiency
d2Index = list(d2.index) # get the index of the transition table for efficiency
def transition(y, y_1):
    if y in d2Index:
        if y_1 in d2cols:
            # return ((d2.loc[tag1][tag2])/(sum(d2[tag2])))
            return d2.loc[y][y_1]
        else:
            # print(3,"oh")
            return 0
    else:
        # print(4, 'oh')
        return 0

print(d) # print the emission table
print(d2) # print the transition table
tagsWithStart = list(d2.columns)
vocab = list(d.columns)
tagsNoStart = list(d2.index)
tagsNoStart.append("START")
scores = []
predictions = [("WORD", "TAG", "PREDICTED TAG")]


'''Function that employs hand crafted rules to deal with unknown words - feature based mappings
These rules were determined both by observation and with the help of a first language speaker'''
def dealWithUNK(word):
    if len(word) > 5 and word[-4:] == "lela":
        if  word[-5] in ['a','e','i','o','u']:
            return "V"
        else:
            return "N"
    if word[-2:] == "ne":
        return "V"
    if word[:2] == "ne":
        return "N"
    if word[0] == "a":
        return "V"
    
    if "-" in word:
        if word[:4] == "nge-":
            return "P"
        if word[0] == "u" or word[0] == "U":
            if str.capitalize(word.split("-")[-1]) == word.split("-")[-1]:
                return "V"
            else:
                return "N"
        
        if str.isdigit(word.split("-")[-1]):
            return "REL"
        if str.capitalize(word.split("-")[-1]) == word.split("-")[-1]:
            return 
    if word[0] == 'e' and str.capitalize(word[1]) == word[1]:
        return "N"
    if not str.isalpha(word):
        return "NUM"
    return None
    

'''Viterbi algorithm. Note that this algorithm is O(n*k^2) where n is the length of the sentence and k is the number of tags.
The number of tags is 74 however this implimentation seems very slow. Could be pandas?? :('''
def viterbi(testSentence, tags):
    posCount = 0 # count the number of POS tags that were correct in a sentence
    backPointer = [{}] # initialize the backpointer
    pi = [{}] # initialize the pi table

    # initialize the pi table and backpointer for the first word 
    for j in range(0,len(tagsNoStart)):
        if tagsNoStart[j] == "START":
            pi[0]["START"] = 1
            backPointer[0][tagsNoStart[j]] = "START"
        else:
            pi[0][tagsNoStart[j]] = 0
            backPointer[0][tagsNoStart[j]] = "START"
    
    # Main viterbi algorithm loop for all words in the sentence
    for i in range(1, len(testSentence)):
        inVocabBool = testSentence[i] in vocab # check if the word is in the vocabulary (done once for efficiency)
        if not inVocabBool:
            tag = dealWithUNK(testSentence[i]) # if the word is not in the vocab, use the rules to determine the tag (done once for efficiency)
        pi.append({}) # initialize the pi table for the next word
        backPointer.append({}) # initialize the backpointer for the next word
        for s in tagsNoStart:
            max = -100000000 # initialize the max value 
            kMax = "" # initialize the tag that maximizes the value
            for k in pi[i-1]:        
                if inVocabBool:  
                    # new = pi[i-1][k] + transition(s,k) + emission(testSentence[i], s) # Used for log probabilities
                    new = pi[i-1][k] * transition(s,k) * emission(testSentence[i], s) 
                else:
                    if k == tag and tag != None:
                        new = pi[i-1][k] * transition(s,k) * 1
                        # new = pi[i-1][k] + transition(s,k) + 0 # Used for log probabilities
                    else:
                        new = pi[i-1][k] * transition(s,k) * emission("UNK", s)
                        # new = pi[i-1][k] + transition(s,k) + emission("UNK", s) # Used for log probabilities
                if new > max: 
                    max = new # update the max value
                    kMax = k # update the tag that maximizes the value
            pi[i][s] = max # update the pi table
            backPointer[i][s] = kMax # update the backpointer
    
    # Uncomment to see the pi table and backpointer in full for each sentence (in a nice table format)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(pd.DataFrame(backPointer).T)    
    #     print(pd.DataFrame(pi).T)

    # Backward step to recover optimal path
    prevSym = "END" # initialize the previous tag used to backtrack
    predTags = ["END"] # initialize the predicted tags that record backtracked tags
    for x in range(len(testSentence)-1, 0 , -1):
        predTags = [backPointer[x][prevSym]] + predTags
        prevSym = backPointer[x][prevSym] 
        
    # Retrieve the predicted tags and the actual tags for accuracy calculation and writing to file
    for x in range(len(predTags)):
        # print(testSentence[x] , predTags[x], tags[x], sep="\t") # uncomment to see results for each sentence in real time
        predictions.append((testSentence[x],tags[x], predTags[x]))
        if predTags[x] == tags[x]:
            posCount += 1  
    
    # Calculate the accuracy for the sentence
    accuracy_score = posCount/len(testSentence)
    print(accuracy_score)
    scores.append(accuracy_score)
    predictions.append(accuracy_score)
        
'''Run each sentence through viterbi algorithm'''
for testSentence in sentences:
    temp = [[ i for i, j in testSentence ],
       [ j for i, j in testSentence ]] # decompose the sentence into words and tags
    sent = temp[0]
    tags = temp[1]
    viterbi(sent, tags) # run the sentence through the viterbi algorithm


with open("testPredictions.txt","w") as f:
    f.write("\n".join(str(ln) for ln in predictions)) # write the predictions to a file

print("AVG SCORE: ", sum(scores)/len(scores)) # print the average accuracy score over all sentences