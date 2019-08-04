import numpy, sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import myPredict as mp

filename = "./text"
raw_text = open(filename).read()
raw_text = raw_text.lower()
seq_in = ""

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100
X = []
Y = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    Y.append(char_to_int[seq_out])
n_patterns = len(X)
print("Total Patterns: ", n_patterns)

X = numpy.reshape(X, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(Y)

model = Sequential()
#model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=8, callbacks=callbacks_list)

model.compile(loss='categorical_crossentropy', optimizer='adam')
start = numpy.random.randint(0, len(X)-1)
pattern = X[start]
print("n", n_vocab)
print("Seed:")
print("\"", ''.join([int_to_char[round(value*n_vocab)] for value in pattern.flatten()]), "\"")
pattern=pattern.flatten().tolist()
for i in range(10):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(x, verbose=0)
    index = numpy.random.choice(len(prediction[0]), p=prediction[0]) # SAMPLE RANDOMLY WITH PROBABILITIES FROM PREDICTOR
    result = int_to_char[index]
    seq_in = [int_to_char[round(value*n_vocab)] for value in pattern]
    pattern.append(index/n_vocab)
    pattern = pattern[1:len(pattern)]
print("".join(seq_in))
print("\nDone.")