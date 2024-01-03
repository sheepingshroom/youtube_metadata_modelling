import pandas as pd

INPUT = "data/testing_unbalanced1.pkl"
OUTPUT = "data/testing_unbalanced.csv"

def csv_to_pickle(input):
    df = pd.read_csv(input)
    df.to_pickle(OUTPUT)

def pickle_to_csv(input):
    df = pd.read_pickle(input)
    df.to_csv(OUTPUT)

pickle_to_csv(INPUT)