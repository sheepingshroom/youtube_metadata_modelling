import pandas as pd
import os

FOLDER_PATH = "data/dataset_10k_subs/channels"
OUTPUT_NAME = "dataset_10k_5.0"

files = [FOLDER_PATH + "/" + x for x in os.listdir(FOLDER_PATH) if x[-4:] == ".csv"]

df = pd.concat(map(pd.read_csv, files), ignore_index=True)

df.to_csv(FOLDER_PATH + "/" + OUTPUT_NAME + ".csv")


