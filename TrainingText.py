import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

#New Imports
import tensorflow as tf
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import RMSprop

print("Preparing textGenerator...")
# filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
print("Type in the filename. Make sure that the file is within the directory.")
print("For example: SimpsonS1-S5.txt")
message = input("")
filepath = message
print("filepath being used: ", filepath)
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #We change everything to lowercase but we don't make use of the uppercases


print("Type in text Slice Dimensions starting with Left...")
print("For example: SimpsonS1-S5.txt uses 2000:200000, so for left type 2000 and for right type 200000")
Left = int(input(""))
print("Type in text Slice Dimensions for Right...")
Right = int(input(""))
text = text[Left:Right]

print("What would you like to call your model?")
print("For example: NameOfModel.model")
modelName = input("")
#Remember it has to be the same size as text (or bigger maybe) as the one we used to test otherwise it may not work
#textgenerator V4 uses: text = text[100:100000]
#textgenerator V5 uses: text = text[1:300000]
#textGeneratorSpongeBobV1  uses: text[1:1000]
#textGeneratorSpongeBobV2 uses: text = text[1000:100000]
#textGeneratorSpongeBobV3 uses: text[1000:8000]
#SimpsonS1-S5 uses: text = text[2000:200000]
#textGenerator_GoTV1 uses: text = text[1000:50000]
characters = sorted(set(text))

#Creates a dictionary that has a character as a key (c) and the index as value (i) and pass the indices as the enumeration of characters and vice versa
#Enumerate assigns 1 number in this set.
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40 #Size of the sequence
STEP_SIZE = 3 #How many characters we gonna shift to the next sentence

# We can try below into Training.PY but we need to use the above parameters, it may be a bit problematic though
# Since the above parameters need to be consistent with the code here and the neural network you'd put elsewhere.
# ---------------------------------------------------------------------------------------------------


sentences = []
next_characters = []
print("Finished collecting training data...")
timeLapse = 0;

# Range 0 to the length of the text. So the beginning of text to the last sequence of the text, with a step size of 3
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH]) #Append the text from i to the sequence length.
    next_characters.append(text[i+SEQ_LENGTH]) #Append the text for the next characters

    #Whenever in a specific sentence, at a specific position does a character occur true or false?
    #15min in
    print("Loading... ", timeLapse)
    timeLapse = timeLapse+1
    x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
    y = np.zeros((len(sentences), len(characters)), dtype=bool)

    #So we iterate through each sentence and assign an index to each sentence
    # Also for each sentence we enumerate every character in the sentence and a
    #
    for i, sentence in enumerate(sentences):
        for t, character in enumerate(sentence):
            x[i, t, char_to_index[character]] = 1 #Take the character in that position and set its value to 1.
        y[i, char_to_index[next_characters[i]]] = 1 #In this sentence, the next character is this one.


print("Working on neural network...")
# Neural Network
model = Sequential()
model.add(LSTM(units=128, input_shape=(SEQ_LENGTH, len(characters)), return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
model.summary()
model.fit(x, y, batch_size=256, epochs=200, verbose=2)
model.save(modelName) #Save the model for later
print("done")