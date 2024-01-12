import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[30000:80000]
characters = sorted(set(text))
'''Now we create a sorted set of all the unique characters that occur in the text. 
In a set no value appears more than once, so this is a good way to filter out the characters. After that we define two structures for converting the values. 
Both are dictionaries that enumerate the characters. In the first one, the characters are the keys and the indices are the values. 
In the second one it is the other way around. Now we can easily convert a character into a unique numerical representation and vice versa. '''
char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))

SEQ_LENGTH =40
STEP_SIZE =3

sentences = []
next_charecters = []
#We iterate through the whole text and gather all sentences and their next character. This is the training data for our neural network.
for i in range(0,len(text)-SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_charecters.append(text[i+SEQ_LENGTH])

#Now we just need to convert it into a numerical format.
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)
'''Wherever a character appears in a certain sentence at a certain position we will set it to a one or a True. 
We have one dimension for the sentences, one dimension for the positions of the characters within the sentences and 
one dimension to specify which character is at this position. '''
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_charecters[i]]] = 1

'''We will use Sequential for our model, Activation, Dense and LSTM for our layers and RMSprop for optimization during the compilation of our model. 
LSTM stands for long-short-term memory and is a type of recurrent neural network layer. It might be called the memory of our model. 
This is crucial, since we are dealing with sequential data. '''
model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

'''Now we compile the model and train it with our training data that we prepared above. 
We choose a batch size of 256 (which you can change if you want) and four epochs. 
This means that our model is going to see the same data four times. '''

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.01))

model.fit(x, y, batch_size=256, epochs=4)

#Our model is now trained but it only outputs the probabilities for the next character.
# We therefore need some additional functions to make our script generate some reasonable text.
#This helper function called sample is copied from the official Keras tutorial.
#Link to the tutorial: https://keras.io/examples/lstm_text_generation/
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
#Now we can get to the final function of our script. The function that generates the final text.
def generate_text(length, temprature):
    start_index = random.randint(0,len(text) - SEQ_LENGTH-1)
    generated =''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0,t, char_to_index[character]]=1

        predictions = model.predict(x, verbose =0)[0]
        next_index = sample(predictions, temprature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence =sentence[1:] + next_character
    return generated
print('------0.2-------------')
print(generate_text(300, 0.2))
print('------0.4-------------')
print(generate_text(300, 0.4))
print('------0.5-------------')
print(generate_text(300, 0.5))
print('------0.6-------------')
print(generate_text(300, 0.6))
print('------0.7-------------')
print(generate_text(300, 0.7))
print('------0.8-------------')
print(generate_text(300, 0.8))



