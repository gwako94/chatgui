import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

wnl = WordNetLemmatizer()


words = []
classes = []
documents = []

ignore_words = ["?", "!"]


data_file = open("intents.json").read()

intents = json.loads(data_file)


for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # Add documents in the corpus
        documents.append((w, intent["tag"]))

        # add tags to the classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatize and lower each word.
words  = [wnl.lemmatize(w.lower()) for w in words if w not in ignore_words]

# sort and remove duplicates
words  = sorted(list(set(words)))


# sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")

# classes = intents
print (len(classes), "classes", classes)

# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
