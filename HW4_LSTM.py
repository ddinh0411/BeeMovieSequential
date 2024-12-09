# loading imports
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer

from random import randint
from pickle import load
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import string

filename = 'beemovie.txt'
seq_len = 100

def load_doc(filename):
  file = open(filename, 'r')
  text = file.read()
  file.close()
  return text

def clean_doc(doc):
  doc = doc.replace('--', ' ')
  tokens = doc.split()
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens

def save_doc(lines, filename):
  data = '\n'.join(lines)
  file = open(filename, 'w')
  file.write(data)
  file.close()

doc = load_doc(filename)
print(doc[:200])

tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = seq_len + 1
sequences = list()

for i in range(length, len(tokens)):
  seq = tokens[i-length:i]
  line = ' '.join(seq)
  sequences.append(line)
print('Total Sequences: %d' % len(sequences))

out_filename = filename[:-4] + '_seq.txt'
save_doc(sequences, out_filename)

# load doc into memory
def load_doc(filename):
  file = open(filename, 'r')
  text = file.read()
  file.close()
  return text

# load
doc = load_doc(out_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1

sequences = np.array(sequences)
sequences.shape
X, Y = sequences[:,:-1], sequences[:,-1]
Y = to_categorical(Y, num_classes=vocab_size)
seq_length = X.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, 100 , input_length = seq_length))
model.add(LSTM(100 , return_sequences=True))
model.add(LSTM(100 , return_sequences=True))
model.add(LSTM(100, dropout= 0.2))
model.add(Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, Y, batch_size=128, epochs=300)

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
  for _ in range(n_words): # generate a fixed number of words
    encoded = tokenizer.texts_to_sequences([in_text])[0] # encode the text as integer
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre') # truncate sequences to a fixed length
    yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1) # predict probabilities for each word
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    in_text += ' ' + out_word # append to input
    result.append(out_word)
  return ' '.join(result)
 
in_filename = 'beemovie_seq.txt' # load cleaned text sequences

doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1


sequence_file = open("sequence_output_LSTM.txt","w")

for i in range(0, 10):
  seed_text = lines[randint(0,len(lines))]
  generated = generate_seq(model, tokenizer, seq_length, seed_text, 100)
  sequence_file.write(generated)
  sequence_file.write("\n")

sequence_file.close()