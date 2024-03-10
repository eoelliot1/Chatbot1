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
#Video 2 content:
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #We change everything to lowercase but we don't make use of the uppercases
text = text[100:100000]
characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40 #Size of the sequence
STEP_SIZE = 3 #How many characters we gonna shift to the next sentence

# We can try below into Training.PY but we need to use the above parameters, it may be a bit problematic though
# Since the above parameters need to be consistent with the code here and the neural network you'd put elsewhere.
# # ---------------------------------------------------------------------------------------------------
# # Prepartion for neural network, might no longer be needed in the code once the neural network is finished
# # So remember to comment out once ur done
#
# sentences = []
# next_characters = []
# print("Finished collecting training data...")
#
# # Range 0 to the length of the text. So the beginning of text to the last sequence of the text, with a step size of 3
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i+SEQ_LENGTH]) #Append the text from i to the sequence length.
#     next_characters.append(text[i+SEQ_LENGTH]) #Append the text for the next characters
#
#     #Whenever in a specific sentence, at a specific position does a character occur true or false?
#     #15min in
#     x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
#     y = np.zeros((len(sentences), len(characters)), dtype=bool)
#
#     #So we iterate through each sentence and assign an index to each sentence
#     # Also for each sentence we enumerate every character in the sentence and a
#     #
#     for i, sentence in enumerate(sentences):
#         for t, character in enumerate(sentence):
#             # print(sentence, end="")
#             x[i, t, char_to_index[character]] = 1 #Take the character in that position and set its value to 1.
#         y[i, char_to_index[next_characters[i]]] = 1 #In this sentence, the next character is this one.
#
# # ---------------------------------------------------------------------------------------------------
# #  Neural network in video
# print("Working on neural network...")
#
# # Videos neural network:
# # -----------------------------------------------------------
# # model = Sequential()
# # model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# # model.add(Dense(len(characters)))
# # model.add(Activation('softmax'))
# # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
# # model.summary()
# # model.fit(x, y, batch_size=256, epochs=4)
# # model.save('textgenerator.model') #Save the model for later
#
# # My neural network:
# # -------------------------------------------------------------
#
# model = Sequential()
# model.add(LSTM(units=128, input_shape=(SEQ_LENGTH, len(characters)), return_sequences=True))
# model.add(LSTM(units=64))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
# model.summary()
# model.fit(x, y, batch_size=256, epochs=4, verbose=2)
# model.save('textgeneratorV2.model') #Save the model for later
#
# # score = model.evaluate(x, y, verbose=2) #Should be some sort of output here, cuz typically it asks for input x and output y, so maybe we need the actual results here go and find it
#
# # Printing Test and Accuracy results
# # print("\nTest score/loss:", score[0])
# # print("Test accuracy:", score[1])
# #Neural network in video
#
# #Once you're done running this, you can actually remove the neural network code and just put down this:
# ---------------------------------------------------------------------------------------------------

model = tf.keras.models.load_model('textgeneratorV1_SpongeBob.model') #Replace with the model you want to generate.

# Copied from keras tutorial
# Takes the prediction of a model and picks 1 character
# High temperature = more creative the sentences is but riskier, low temperature = less creative
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Copying the first 'SEQ_LENGTH' number of characters from text.
# If you want text generated completely from scrach then remove first 'SEQ_LENGTH' of characters
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1) # We do - SEQ_LENGTH so that we have enough characters. we do -1 since we dealing with indicies
    generated = '' # The generated text
    sentence = text[start_index: start_index + SEQ_LENGTH] #So we take only the start_index from the (start_index + SEQ_LENGTH)
    generated += sentence #Add the text to generated
    for i in range(length):
        #So like how we did in the LSTM, we just index each an every character.
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print("Generating Text:")

print("------------------------temp 0.2")
print(generate_text(300, 0.2))

print("------------------------temp 0.4")
print(generate_text(300, 0.4))

print("------------------------temp 0.6")
print(generate_text(300, 0.6))

print("------------------------temp 0.8")
print(generate_text(300, 0.8))

print("------------------------temp 1")
print(generate_text(300, 1))

