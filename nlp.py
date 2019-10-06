from urllib.request import urlopen
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import re

def simplify(w):
    # Removing extra punctuation
    w = w.replace("-", "").replace("~","")
    
    # Replacing numbers with # sign
    w = re.sub('\d', '#', w)
    
    # Changing some endings
    if len(w) > 3 and w[-2:] in set(["ed", "er","ly"]):
        return [w[:-2], w[-2:]]
    elif len(w) > 4 and w[-3:] in set(["ing","'re"]):
        return [w[:-3], w[-3:]]
    return [w]

# Opening a text file of subtitles from VlogBrothers
txt = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab2-nlp/master/vlogbrothers.txt").read().decode('ascii').split("\n")
#print(txt[1])
#print("The dataset contains {} vlogbrothers scripts".format(len(txt)))
everything = set([w for s in txt for w in s.split()])
#print("and {} lexical types".format(len(everything)))

# Tokenizing the data
nlp = spacy.load("en_core_web_sm", disable = ["parser", "tagger", "ner", "textcat"])
txt = [nlp(s) for s in txt]

# Marking the beginning and end of each script
txt = [["<s>"] + [str(w) for w in s] + ["</s>"] for s in txt]

# Separating the data into training and validation
train = txt[:-5]
valid = txt[-5:]

# Flattening the lists into one long string and removing extra whitespace
train = [w for s in train for w in s if not w.isspace()]
valid = [w for s in valid for w in s if not w.isspace()]
#print("The training dataset contains {} lexical types".format(len(set(train))))
#print("The training dataset contains {} lexical tokens".format(len(train)))

# Cleaning the data
train_clean, valid_clean = [], []
for w in train:
    for piece in simplify(w):
        train_clean.append(piece)
for w in valid:
    for piece in simplify(w):
        valid_clean.append(piece)

# Checking the size of the dataset
print("{} lexical types".format(len(set(train_clean))))
print("{} lexical tokens".format(len(train_clean)))

# Counting the frequencies of words
"""
counts_clean = Counter(train_clean)
train_unk = [w if counts_clean[w] > 1 else "unk" for w in train_clean]
valid_unk = [w if w in counts_clean and counts_clean[w] > 1 \
               else "unk" for w in valid_clean]

counts = Counter(train_unk)
frequencies = [0] * 8
for w in counts:
  if counts[w] >= 128:
    frequencies[0] += 1
  elif counts[w] >= 64:
    frequencies[1] += 1
  elif counts[w] >= 32:
    frequencies[2] += 1
  elif counts[w] >= 16:
    frequencies[3] += 1
  elif counts[w] >= 8:
    frequencies[4] += 1
  elif counts[w] >= 4:
    frequencies[5] += 1
  elif counts[w] >= 2:
    frequencies[6] += 1
  else:
    frequencies[7] += 1

# Plotting the distributions
f, a = plt.subplots(1, 1, figsize = (10,5))
a.set(xlabel = "Lexical types occuring more than n times", 
      ylabel = "Number of lexical types")

labels = [128, 64, 32, 16, 8, 4, 2, 1]
_ = sns.barplot(labels, frequencies, ax = a, order = labels)
plt.show()
"""