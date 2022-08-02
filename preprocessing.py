import numpy as np 
import sys 

path = sys.argv[1]

with open(path, 'r') as f:
    lines = f.readlines()

    # Remove the first line
    lines.pop(0)
count = 0
newLines = [('^','START')]
sentences = []
for l in lines:
    word = l.split('#')[0]
    tag = l.split('#')[1].strip()
    # print(word, tag)
    if tag != 'PUNCT' and tag != '':
        newLines.append((word,tag))
    elif tag == '':
        newLines.append(('$','END'))
        sentences.append(newLines)
        newLines = [('^','START')]
    count += 1

sentences = np.array(sentences, dtype=object)
np.save('sentences.npy', sentences)