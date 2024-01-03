import preprocessor
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import utils
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.regularizers import l2
import pandas as pd
import os
import numpy as np
import pickle

INPUT_FOLDER = "data"
TOKENIZER_PATH = "data/tokenizer_final.pkl"

INPUT_PATH = {
    "train": os.path.join(INPUT_FOLDER, "training_final.pkl"),
    "val": os.path.join(INPUT_FOLDER, "validation_final.pkl")
}

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

df = {
    "train": pd.read_pickle(INPUT_PATH["train"]),
    "val": pd.read_pickle(INPUT_PATH["val"])
}

sequences = {
    "train": np.array([np.array(val) for val in df["train"].sequences]),
    "val": np.array([np.array(val) for val in df["val"].sequences])
}

vocab_length = len(tokenizer.word_index) + 1
max_length = len(df["train"].sequences[0])

classification = {
    "train": df["train"].classification.to_numpy(),
    "val": df["val"].classification.to_numpy()
}

classification["train"] = utils.to_categorical(classification["train"], 3)
classification["val"] = utils.to_categorical(classification["val"], 3)

print(max_length)
print(vocab_length)
print(sequences["train"].shape)

model = keras.models.Sequential()
model.add(layers.Embedding(vocab_length, 32, input_length=max_length))

model.add(layers.LSTM(64, dropout=0.2, return_sequences=True))
model.add(layers.LSTM(64, dropout=0.2, return_sequences=True))
model.add(layers.LSTM(64, dropout=0.2, return_sequences=True))
model.add(layers.LSTM(64, dropout=0.2))
#model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
model.add(layers.Dense(3, activation="softmax"))

model.summary()

loss = keras.losses.CategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.legacy.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

history = model.fit(sequences["train"], classification["train"], epochs=3, validation_data=(sequences["val"], classification["val"]), verbose=1)

model.save("stacked_lstm_final.keras")

with open('stacked_lstm_final_hist.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)