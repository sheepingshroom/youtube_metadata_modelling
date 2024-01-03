import requests 
from PIL import Image 
import os
import pandas as pd
import time
import math

OUTPUT_PATH = "data/video_dataset/thumbnails" 
INPUT_CSV = "data/video_dataset/dataset_10k_lang_5.0.csv"

df = pd.read_csv(INPUT_CSV)

scanned_list = os.listdir(OUTPUT_PATH)

for i, row in df.iterrows():

    filename = f"{i}.jpg"

    if filename not in scanned_list:
        url = df.at[i, "thumbnail"]
        if url == url:
            if url[:4] == "http":
                data = requests.get(url).content
                with open(os.path.join(OUTPUT_PATH,f"{i}.jpg"), "wb") as f:
                    f.write(data) 
                    print(f"writing filename: {filename}")
                    f.close() 
                time.sleep(0.02)
    else:
        print("Skipping")

