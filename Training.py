import random
import numpy
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt')  # --Run this line of code if punkt is not installed.
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimzers import SGD --This is currently giving error: from tensorflow.keras.optimzers import SGD  ModuleNotFoundError: No module named 'tensorflow.keras.optimzers'

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ','] #Letters we don't consider

#Iterates through all the intents in the intent document.
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #Splits texts into individual words.
        words.extend(word_list) #Used to be words.append(word_list)
        documents.append((word_list, intent['tag'])) #So we know what in word_list belongs to what intent tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

           # print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #Set eliminate duplicates and sorts turns it into a list and sorts it.

print('Done words')

classes = sorted(set(classes)) # Sort out classes as well
print('1')
#Save words into a file
pickle.dump(words, open('words.pkl', 'wb'))
#Save classes into a file
pickle.dump(classes, open('classes.pkl', 'wb'))
print('2')
# Values need to be numerical for neural network compatability
training = []
output_empty = [0] * len(classes)

for documents in documents:
    bag = [] #Empty bag of words
    word_patterns = documents[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

# Append word into bag if it is in word_patterns (by putting it in array1)
# We want to see if each word occurs in the word_pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty) #Copying the list not type casting
    output_row[classes.index(documents[1])] = 1 #We want to know the class at index1 and set this index in the ouput row to 1
    training.append(bag + output_row)

random.shuffle(training) #Shuffle training data
training = np.array(training)

print('Working on assigning data to x and y...')
# Dimensions for training data, the features (train_x) and labels (train_y)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

#Create your Neuaral network
print('Working on creating Neural network...')
model = tf.keras.Sequential()

# My neural network
model.add(tf.keras.layers.Dense(units=128, input_shape=(len(train_x[0]),), activation='relu')) #Determines the filters, Kernel size, and the shape of the input for the neural network.
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=len(train_y[0]), activation='softmax'))
# My neural network

sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=2)

# Summary of the model.
model.summary()

#Evaluating model on testing samples.
score = model.evaluate(train_x, train_y, verbose=2) #Should be some sort of output here, cuz typically it asks for input x and output y, so maybe we need the actual results here go and find it

# Printing Test and Accuracy results
print("\nTest score/loss:", score[0])
print("Test accuracy:", score[1])


model.save('chatbot_model.h7', hist)
print('Done')