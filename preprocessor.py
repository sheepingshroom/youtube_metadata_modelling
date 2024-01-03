import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import string
from collections import Counter
from nltk.tokenize import word_tokenize
import os
import pickle
import random
import logging

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This gets logged to a file')

DATA_FOLDER = "data"
INPUT_PATH = { #path of non-processed data
    "unbal": os.path.join(DATA_FOLDER, "model_dataset_unbalanced.csv"), 
    "bal": os.path.join(DATA_FOLDER, "model_dataset_balanced.csv"),
    "universe": os.path.join(DATA_FOLDER, "model_dataset_youniverse_micro1.csv")
}

OUTPUT_PATH = {
    "train_unbal": os.path.join(DATA_FOLDER,"training_unbalanced.pkl"),
    "val_unbal": os.path.join(DATA_FOLDER,"validation_unbalanced.pkl"),
    "train_bal": os.path.join(DATA_FOLDER,"training_balanced.pkl"),
    "val_bal": os.path.join(DATA_FOLDER,"validation_balanced.pkl"),  
    "tokenizer": os.path.join(DATA_FOLDER,"tokenizer.pkl"),  
    "train_universe": os.path.join(DATA_FOLDER,"training_universe4.pkl"),
    "val_universe": os.path.join(DATA_FOLDER,"validation_universe4.pkl"),
    "test_universe": os.path.join(DATA_FOLDER,"testing_universe4.pkl"),
    "tokenizer_universe": os.path.join(DATA_FOLDER,"tokenizer_universe.pkl")
}

TRAINING_RATIO = 0.8
TEST_RATIO = 0.1
SEQUENCE_MAX_LENGTH = 25

df = {
    "unbal": pd.read_csv(INPUT_PATH["unbal"]),
    "bal": pd.read_csv(INPUT_PATH["bal"]),
    "universe": pd.read_csv(INPUT_PATH["universe"])
}

def parse_text(text):
    print(text)
    if (type(text) == str):
        pass
    else:
        text = str(text)

    text = text.lower() #lowercase conversion
    text = word_tokenize(text)
    return text

def remove_tokens(text, blacklist):
    text = [x for x in text if x not in blacklist]
    print("tokens removed")
    return text

def remove_unknown_words(text, counter):
    text = [x for x in text if counter[x]]
    return text

def count_words(text_col):
    count = Counter()
    for text in text_col:
        for word in text:
            count[word] += 1
    return count

def shuffle_by_channel(df):
    ids = df["channel_id"].unique()
    random.shuffle(ids)
    df = df.set_index("channel_id").loc[ids].reset_index()
    print(df.head(10))
    return df

def generate_dataset(df):

    #parse text 
    df["title"] = df.apply(lambda row: parse_text(row["title"]), axis = 1)

    #split into training and validation set
    val_index = int(df.shape[0] * TRAINING_RATIO)
    split = {
        "train": df[:val_index],
        "val": df[val_index:]
    }

    split["val"] = split["val"].reset_index() #reset index of val

    #count each word
    counter = count_words(split["train"]["title"])
    print(counter.most_common(20))
    print(len(counter))
    print(df.head())

    # remove words that only appear once
    blacklist = set()
    for word in list(counter):
        if word == 1:
            print("deleting word")
            del counter[word]
            blacklist.add(word)

    #remove words in titles
    split["train"]["title"] = df.apply(lambda row: remove_tokens(row["title"], blacklist), axis = 1)    
    split["val"]["title"] = df.apply(lambda row: remove_unknown_words(row["title"], counter), axis = 1)   

    print(f"Training Length: {len(split['train'])} Validation Length: {len(split['val'])}")

    #keras tokenization
    tokenizer = Tokenizer(len(counter))
    tokenizer.fit_on_texts(split["train"]["title"])
    print(tokenizer.word_index)
    split["tokenizer"] = tokenizer

    #save tokenizer
    with open(OUTPUT_PATH["tokenizer_universe"], "wb") as handle:
        pickle.dump(split["tokenizer"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = {
        "train": tokenizer.texts_to_sequences(split["train"]["title"]),
        "val": tokenizer.texts_to_sequences(split["val"]["title"])
    }

    print(split["train"]["title"][10:15])
    print(sequences["train"][10:15])

    #padding
    sequences["train"] = pad_sequences(sequences["train"], maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")
    sequences["val"] = pad_sequences(sequences["val"], maxlen=SEQUENCE_MAX_LENGTH, padding="post", truncating="post")

    print("padding done")

    #placeholder values before iterating
    split["train"]["sequences"] = split["train"]["title"] 
    split["val"]["sequences"] = split["val"]["title"]

    for i, row in split["train"].iterrows():
        split["train"].at[i, "sequences"] = sequences["train"][i]

    print("train set done")

    for i, row in split["val"].iterrows():
        split["val"].at[i, "sequences"] = sequences["val"][i]

    print("finished")

    return split

if __name__ == "__main__":
    test_index = int(df["universe"].shape[0] * 1-TEST_RATIO)
    df["universe"][:test_index].to_csv(OUTPUT_PATH["test_universe"], chunksize=1000000)
    output = generate_dataset(df["universe"][:test_index])
    output["train"].to_pickle(OUTPUT_PATH["train_universe"])
    print("saved training set")
    output["val"].to_pickle(OUTPUT_PATH["val_universe"])
    print("saved val set")


