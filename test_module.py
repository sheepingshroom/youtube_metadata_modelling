import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras import Model
from keras.preprocessing.sequence import pad_sequences
import preprocessor
import pickle
import numpy

#model to test
MODEL_PATH = "models/lstm_cnn_final.keras"
TOKENIZER_PATH = "data/tokenizer_final.pkl"
SEQUENCE_MAX_LENGTH = 25

model = tf.keras.saving.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

def decode(text, tokenizer, max_length):
    text = preprocessor.parse_text(text)
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text, maxlen=max_length, padding="post", truncating="post")
    return text
    
def predict_proba(text):
    title = decode(text, tokenizer, SEQUENCE_MAX_LENGTH)
    result = model.predict_on_batch(title)
    return result

def predict_proba_array(texts, m = model):
    titles = []
    for text in texts:
        titles.append(decode(text, tokenizer, SEQUENCE_MAX_LENGTH))
    predicts = []
    for title in titles:
        predicts.append(m.predict_on_batch(title)[0]) 
    return numpy.array(predicts)

def predict_proba_list(texts, m = model):
    titles = []
    for text in texts:
        titles.append(decode(text, tokenizer, SEQUENCE_MAX_LENGTH))
    predicts = []
    for title in titles:
        predicts.append(m.predict_on_batch(title)[0]) 
    return predicts

def model_info():
    print(model.summary)



