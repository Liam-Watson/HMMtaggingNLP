import numpy as np
import pandas as pd

sentences = np.load('sentences.npy', allow_pickle=True)

d = {}
d2 = {}
count = 1
for s in sentences:
    for t in s:
        word, tag = t
        if tag in d:
            if word in d[tag]:
                d[tag][word] += 1
            else:
                d[tag][word] = 1
        else:
            d[tag] = {word:1}
        if "$" != word:
            wP1, tagP1 = s[count]
            # print(tag, tagP1)
            if tagP1 in d2:
                if tag in d2[tagP1]:
                    d2[tagP1][tag] += 1
                else:
                    d2[tagP1][tag] = 1
            else:
                d2[tagP1] = {tag:1}
            count+=1
    count = 1


print(d2["END"])

d = pd.DataFrame(d).T.add(1, level=1, fill_value=0)

d2 = pd.DataFrame(d2).T.add(1, level=1, fill_value=0)

print(d.columns)
# print(d2)
# df1.to_excel("d.xlsx")
# df2.to_excel("d2.xlsx")

def emmision(word, tag):
    if tag in list(d.index):
        if word in list(d.columns):
            return -np.log((d.loc[tag][word])/(sum(d.loc[tag])+len(d.columns)))
        else:
            # print("1oh", word, tag)
            return 0
            
    else:
        # print("2oh")
        return 0


def transition(tag1, tag2):
    if tag2 in list(d2.columns):
        if tag1 in list(d2.index):
            return -np.log((d2.loc[tag1][tag2])/(sum(d2[tag2]) + len(d2.columns)))
        else:
            # print(3,"oh")
            return 0
    else:
        # print(4, 'oh')
        return 0


backPointer = {}
y = list(d2.keys())
testSent = "^ Injongo ye-website yaseMzantsi Afrika kukuvelisa umthombo omnye wenkcazelo malunga neenkonzo ezinikwa ngurhulumente waseMzantsi Afrika $".split()

for x in y:
    for a in testSent:
        if x not in backPointer:
            backPointer[x] = {a: 0.0}
        else:
            backPointer[x][a] = 0.0

pi = [{}]
backPointer = {}
def viterbi():
    for j in range(0,len(y)):
        if y[j] == "START":
            pi[0]["START"] = 1
        else:
            pi[0][y[j]] = 0
    for i in range(1, len(testSent)):
        max = 0
        pi.append({})
        for s in y:
            for k in y:
                new = pi[i-1][k] + transition(s,k) + emmision(testSent[i], s)
                if new > max:
                    max = new
            pi[i][s] = max
        print(i)
        
count = 1

viterbi()
pi2 = pd.DataFrame(pi).T
print(pi2)

for i in range(len(pi)):
    maxx = 0
    jMax = 0
    for j in pi[i]:
        if pi[i][j] > maxx:
            maxx = pi[i][j]
            jMax = j
    print(testSent[i], jMax)
        



