import sys
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Dense


def build_model(input_shape, output):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dense(output, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop())
    
    return model

def process_data(text, chars, char_indices, maxlen=200, step=1):
    sentences = []
    next_chars = []
    for i in range(0, len(text)-maxlen, step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])
    
    # Building one-hot encoding
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    
    return x, y, sentences, next_chars

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        
    return data

# This peice of code adds a little bit of noise by not always prediciting what the model things is the best thing.
# This makes things interesting and helps us understand what different values are being outputted by the model.
def sample(preds, temperature=1.0):
    '''
    1. This peice of code adds a little bit of noise by not always prediciting what the model
       things is the best thing.
    2. This makes things interesting and helps us understand what different values are being outputted 
       by the model.
    '''
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Let's add a callback after each epoch
class SampleText(tf.keras.callbacks.Callback):
    def __init__(self, text, chars, char_indices, indices_char, maxlen):
        self.text = text
        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.maxlen = maxlen
    
    def on_epoch_end(self, batch, logs={}):
        start_idx = np.random.randint(0, len(self.text)-self.maxlen-1)
        # diversity makes the most obvious prediction values smaller
        # More the diversity -> interesting are the outputs(not necessarily what you want).
        # Smaller the value of diversity, outputs would be as expected., try diversity 1.2
        print()
        for diversity in [0.5]:
            print(f'diversity: {diversity}')
            generated = ['']
            sentence = self.text[start_idx:start_idx+self.maxlen]
            #generated += sentence
            
            # sys.out
            # print(generated)
            
            # encoding the input
            idx = 0
            for i in range(200):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.
                # making predictions        
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_idx = sample(preds, diversity)
                next_char = self.indices_char[next_idx]
                
                if next_char!='\n':
                    generated[idx] += next_char 
                else:
                    if generated[idx]!='':
                        generated.append('')
                        idx += 1
                
                sentence = sentence[1:] + next_char
                # sys.out
                # print(next_char, end='')
            print(f'Generated Text : {generated}')
        
        print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', type=str, required=True)
    # Extract the arguments
    args = parser.parse_args()
    file_path = args.train_file
    
    if file_path.split('.')[-1]!="txt":
        print('Expects a txt file!!!')
        sys.exit(0)
    
    # parameter list
    maxlen = 200
    step = 3
    batch_size = 256
    
    # Read raw data and generate vocabulary
    text = read_file(file_path)
    chars = sorted(set(text))
    
    # Assigning indices to characters and vice versa for one-hot encoding
    char_indices = {c:i for i, c in enumerate(chars)}
    indices_char = {i:c for i, c in enumerate(chars)}
    
    # Prepare the data
    x, y, sentences, next_chars = process_data(text, chars, char_indices, maxlen, step)
    
    # build model
    model = build_model(input_shape, output)
    
    model.fit(x, y, batch_size=batch_size,
          epochs=150,
          callbacks=[SampleText()])
    
    
