# from operator import index
import numpy as np
import pandas as pd
import sys

sentences = np.load('train.npy', allow_pickle=True) # read in the train sentences

smoothing = sys.argv[1] # choose the smoothing method


d = {} # create the emission table
d2 = {} # create the transition table
count = 1 # used to keep track of tag_{k} and tag_{k-1}
for s in sentences: # for each sentence
    for t in s: # for each word and tag in the sentence
        word, tag = t
        # Calculate the emission table counts
        if tag in d:
            if word in d[tag]:
                d[tag][word] += 1
            else:
                d[tag][word] = 1
        else:
            d[tag] = {word:1}
        # Calculate the transition table counts
        if "$" != word:
            wP1, tagP1 = s[count]
            if tagP1 in d2:
                if tag in d2[tagP1]:
                    d2[tagP1][tag] += 1
                else:
                    d2[tagP1][tag] = 1
            else:
                d2[tagP1] = {tag:1}
            count+=1
    count = 1

# Apply initial smoothing step to remove NaNs from tables 
if smoothing == "laplace": # Laplace smoothing
    df = pd.DataFrame(d).T.add(1, level=1, fill_value=0)
    df2 = pd.DataFrame(d2).T.add(1, level=1, fill_value=0)
elif smoothing == "add_k": # Add-k smoothing
    k = 0.5
    df = pd.DataFrame(d).T.add(k, level=1, fill_value=0)
    df2 = pd.DataFrame(d2).T.add(k, level=1, fill_value=0)
elif smoothing == "AD": # Absolute Discounting 
    df = pd.DataFrame(d).T.add(0, level=1, fill_value=0)
    df2 = pd.DataFrame(d2).T.add(0, level=1, fill_value=0)
elif smoothing == "GT": # Good Turing
    df = pd.DataFrame(d).T.add(0, level=1, fill_value=0.1)
    df2 = pd.DataFrame(d2).T.add(0, level=1, fill_value=0.1)


if smoothing == "AD": # Absolute Discounting
    print("Calculating AD emission probabilities...")
    discount = 5 # Variable discount parameter which may be tuned for better results
    redistribute = 0
    for row in df.index:
        for col in df.columns:
            if df.loc[row][col] >= discount:
                df.loc[row][col] -= discount
                redistribute += discount
            else:
                redistribute += df.loc[row][col]
                df.loc[row][col] = 0
        v = (df.loc[row] == 0).sum()
        for col in df.columns:
            if df.loc[row][col] == 0:
                df.loc[row][col] = (redistribute/v)
        redistribute = 0
    print("Calculating AD transition probabilities...")
    for col in df2.columns:
        for row in df2.index:
            if df2[col][row] >= discount:
                df2[col][row] -= discount
                redistribute += discount
            else:
                redistribute += df2[col][row]
                df2[col][row] = 0
        v = (df2[col] == 0).sum()
        for row in df2.index:
            if df2[col][row] == 0:
                df2[col][row] = (redistribute/v)
        redistribute = 0
elif smoothing == "GT": # Good Turing
    print("Calculating Good-Turing emission probabilities...")
    for row in df.index:
        largest = df.loc[row].sort_values(inplace=False)
        value_counts = df.loc[row].value_counts() 
        for col in df.columns:
            valueCurrent = df.loc[row][col]
            curIndex = (largest.index).get_loc(col)
            N_k = largest[col]
            N_km1 = largest.iloc[curIndex-1]
            k = value_counts[valueCurrent]
            df.loc[row][col] = (k*N_k)/N_km1
    print("Calculating Good-Turing transition probabilities...")
    for col in df2.columns:
        largest = df2[col].sort_values(inplace=False)
        value_counts = df2[col].value_counts()
        for row in df2.index:
            valueCurrent = df2[col][row]
            curIndex = (largest.index).get_loc(row)
            N_k = largest[row]
            N_km1 = largest.iloc[curIndex-1]
            k = value_counts[valueCurrent]
            df2[col][row] = (k*N_k)/N_km1

print(df) # print the emission table counts after smoothing
print(df2) # print the transition table counts after smoothing

# if smoothing == "laplace" or smoothing == "add_k":
print("Calculating emission probabilities...")
for row in df.index:
    # df.loc[row] = np.log(df.loc[row]/sum(df.loc[row])) # used for log space calculations
    df.loc[row] = (df.loc[row]/sum(df.loc[row])) # Calculate the emission probabilities by normalization done here for efficiency

print("Calculating transition probabilities...")
for col in df2.columns:
    # df2[col] = np.log(df2[col]/sum(df2[col])) # used for log space calculations
    df2[col] = (df2[col]/sum(df2[col])) # Calculate the transition probabilities by normalization done here for efficiency

print(df) # print the emission table probabilities after normalization
print(df2) # print the transition table probabilities after normalization

# Used to check that normalisation is correct and the probabilities sum to ~1 (floating point errors may say otherwise)
# for row in df.index:
#     print(sum(df.loc[row]))

# for col in df2.columns:
#     print(sum(df2[col]))

df.to_pickle("dataTemp/emissionTable_" + smoothing + ".pkl") # Save the emission table to a pickle file
df2.to_pickle("dataTemp/transitionTable_" + smoothing + ".pkl") # Save the transition table to a pickle file

