from langdetect import detect_langs
from ftlangdetect import detect
import pandas as pd

CSV_PATH = "data/dataset_10k_subs/dataset_10k_5.0.csv"

def DetectLangs():
    df = pd.read_csv(CSV_PATH)
    lang_list = []
    for index in df.index:
        title = str(df["title"][index])
        if "\n" not in title:
            lang_list.append(detect(title)["lang"])
            print(f"Detecting Lang from title {title}")
        else:
            lang_list.append("Unknown")


    df["language"] = lang_list
    return df

DetectLangs().to_csv("data/dataset_10k_subs/dataset_10k_lang_5.0.csv")