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


# create training data
training = []

 # create an empty array for the output
output_array = [0] * len(classes)

for doc in documents:

    # initialize our bag of words
    bag = []

    # list of tokenized word for the patterns
    pattern_words = doc[0]
    
    # lematize each word - create a base word in attempt to represent related words

    pattern_words = [wnl.lemmatize(word.lower()) for word in pattern_words]


    # create bag of word array with 1 if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)


    # output is 0 for every tag and 1 for current tag in each pattern

    output_row = list(output_array)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle features and turn into np array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")