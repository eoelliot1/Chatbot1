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
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.optimizers import RMSprop

# Web Imports
import customtkinter
# import tkinter
#
# root = tkinter.Tk()
# label = tkinter.label(root, )
# label.pack()


# ---------------------

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h7')
# ---------- Manual methods

class Gui(customtkinter.CTk):
    # --- UI stuff
    # Works but code wont run until ui is deleted and the commands happen immediatley
    # https://stackoverflow.com/questions/8689964/why-do-some-functions-have-underscores-before-and-after-the-function-name explains that weird convention of the __init__
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("green")

    root = customtkinter.CTk()
    root.geometry("750x450")

    frame = customtkinter.CTkFrame(master=root)
    frame.pack(pady=30, padx=60, fill="both", expand=True)

    label = customtkinter.CTkLabel(master=frame, text="Generate Text", font=("Roboto", 24) ) # adding it to the frame
    label.pack(pady=12, padx=10)

    entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="love")
    entry1.pack(pady=12, padx=10)

    entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="love2")
    entry2.pack(pady=20, padx=15)

    button = customtkinter.CTkButton(master=frame, text="GenerateSpongeBob", command=print("Hello"))
    button.pack(pady=12, padx=10)

    checkbox = customtkinter.CTkCheckBox(master=frame, text="Remember me")
    checkbox.pack(pady=12, padx=10)

    root.mainloop()
    # --- UI stuff


def generateSpongeBob():
    # generateSpongebob
    filepath = "SpongeBobS1&S2.txt"
    option3 = False
    # NeuralNetwork Model for SpongeBob
    model2 = tf.keras.models.load_model('textgeneratorV2_SpongeBob.model')
    print("Reading: SpongeBobS1&S2.txt")

    text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #We change everything to lowercase but we don't make use of the uppercases
    text = text[1000:100000]

    characters = sorted(set(text))
    char_to_index = dict((c, i) for i, c in enumerate(characters))
    index_to_char = dict((i, c) for i, c in enumerate(characters))

    SEQ_LENGTH = 40 #Size of the sequence
    STEP_SIZE = 3 #How many characters we gonna shift to the next sentence
    length = 300 #LengthOfText to generate.
    temperature = 2 #Temperature of generated text

    # GENERATION OF TEXT --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1) # We do - SEQ_LENGTH so that we have enough characters. we do -1 since we dealing with indicies
    generated = '' # The generated text
    sentence = text[start_index: start_index + SEQ_LENGTH] #So we take only the start_index from the (start_index + SEQ_LENGTH)
    generated += sentence #Add the text to generated
    for i in range(length):
        #So like how we did in the LSTM, we just index each an every character.
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

    #So we get the predictions fron the model and we put these predictions into the sample function for the next index.
        predictions = model2.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    print(generated)

#Methods used in interpreting to text inputs and responding to text inputs.
# ---------------------------------------------------------------------------------------------------------------------
# Cleanin up decendants
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#  getting Bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
# Predicting class based on the sentence

# So we create a bag of the words
# We add a certain error threshold to
def predict_class(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    print(res)
    ERROR_THRESHOLD = 0.25 #We're using softmax 30:20 he explains
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in result:
        # print("class: " + classes[r[0]])
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#Methods used in generating text
# ---------------------------------------------------------------------------------------------------------------------
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

        #So we get the predictions fron the model and we put these predictions into the sample function for the next index.
        predictions = model2.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print("Bot running now:")
print("Type: '/GenerateText' to generate a pre-implemented transcript")
print("Type: '/GenerateCustom' to generate a customly built transcript")
print("Note: For customly built transcript you will require a neural network model, a transcript file, and the text Dimensions.")

# Print all the classes
i = 0
for cc in classes:
    print(i)
    print(classes[i])
    i = i + 1


while True:



    message = input("")
    if message == "/GenerateText":

        option1 = True
        option2 = True
        option3 = True

        print("Use preset text array dimensions? Type: Yes/No (RECOMMEND)")
        message = input("")
        if message == "Yes":
            option3 = False
        elif message == "No":
            option3 = True

        # Selection for model
        while option1 == True:
            print("What type of textGeneration would you prefer:")
            print("SpongeBob: SpongeBobS1&S2.txt")
            print("Shakespeare: shakepeare.txt")
            print("Simpson: SimpsonS1-S5.txt")
            message = input("")


            if message == "Shakespeare":
                # Generation for Shakespeare
                filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
                # NeuralNetwork Model for SpongeBob
                model2 = tf.keras.models.load_model('textgeneratorV4.model')
                if option3 == False:
                    left = 100
                    right = 100000
                print("Reading: shakespeare.txt")
                option1 = False

            elif message == "SpongeBob":
                # generateSpongebob
                filepath = "SpongeBobS1&S2.txt"
                # NeuralNetwork Model for SpongeBob
                model2 = tf.keras.models.load_model('textgeneratorV2_SpongeBob.model')
                if option3 == False:
                    left = 1000
                    right = 100000
                print("Reading: SpongeBobS1&S2.txt")
                option1 = False

            elif message == "Simpson":
                # generateSpongebob
                filepath = "SimpsonS1-S5.txt"
                # NeuralNetwork Model for SpongeBob
                model2 = tf.keras.models.load_model('textgenerator_SimpsonV2')
                if option3 == False:
                    left = 2000
                    right = 200000
                print("Reading: SimpsonS1-S5.txt")
                option1 = False


        # Selection for text array dimension
        while option3 == True:
            print("State text dimensions for left.")
            print("For example: SpongeBobV2 uses text[1000:100000] so type in 1000(Left), and then next type 100000(Right).")
            left = int(input(""))

            print("State text dimensions for right")
            right = int(input(""))
            option3 = False

        print("Preparing neural network using selected options...")

        text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #We change everything to lowercase but we don't make use of the uppercases
        text = text[left:right]

        characters = sorted(set(text))
        char_to_index = dict((c, i) for i, c in enumerate(characters))
        index_to_char = dict((i, c) for i, c in enumerate(characters))

        SEQ_LENGTH = 40 #Size of the sequence
        STEP_SIZE = 3 #How many characters we gonna shift to the next sentence

        print("Generating Text:")

        print("------------------------temp 0.1")
        print(generate_text(300, 0.2))

        print("------------------------temp 0.2")
        print(generate_text(300, 0.2))

        print("------------------------temp 0.25")
        print(generate_text(300, 0.2))

        print("------------------------temp 0.4")
        print(generate_text(300, 0.4))

        print("------------------------temp 0.6")
        print(generate_text(300, 0.6))

        print("------------------------temp 0.8")
        print(generate_text(300, 0.8))

        print("------------------------temp 1")
        print(generate_text(300, 1))

    elif message == "/GenerateCustom":
        print("Type the name of the transcript file. Make sure the file is in this current repsoitory.")
        filepath = input("")
        text = open(filepath, 'rb').read().decode(encoding='utf-8').lower() #We change everything to lowercase but we don't make use of the uppercases
        print('You said: ', filepath)

        print("Type the name of the neural network model. Make sure the file is in this current repsoitory.")
        model = input("")
        model2 = tf.keras.models.load_model(model)
        print('You said: ', model)

        print("Type the left text Dimensions")
        left = int(input(""))
        print("Type the right text Dimension")
        right = int(input(""))
        text = text[left:right]
        print('You said: ', 'Text[', left, ':', right, ']')

        print("Generating text...")
        characters = sorted(set(text))
        char_to_index = dict((c, i) for i, c in enumerate(characters))
        index_to_char = dict((i, c) for i, c in enumerate(characters))

        SEQ_LENGTH = 40 #Size of the sequence
        STEP_SIZE = 3 #How many characters we gonna shift to the next sentence

        print("Generating Text:")

        print("------------------------temp 0.1")
        print(generate_text(300, 0.1))

        print("------------------------temp 0.2")
        print(generate_text(300, 0.2))

        print("------------------------temp 0.25")
        print(generate_text(300, 0.25))

        print("------------------------temp 0.4")
        print(generate_text(300, 0.4))

        print("------------------------temp 0.6")
        print(generate_text(300, 0.6))

        print("------------------------temp 0.8")
        print(generate_text(300, 0.8))

        print("------------------------temp 1")
        print(generate_text(300, 1))

    elif message == "Pop":
        print("Snap crackle pop!")

# Text-based interface for printing responses.
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)

# Getting a response

#Chat bot seems to be good at predicing longer sentence things...
#Performance is eh but it does somewhat work