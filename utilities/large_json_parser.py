import jsonlines
import pandas as pd

PATH = "data/youniverse/yt_metadata_en.jsonl"

data_dict = {
    "view_count": [],
    "title": [],
    "channel_id": [],
}

with jsonlines.open(PATH) as f:
    count = 0
    for line in f.iter():
        count += 1
        data_dict["view_count"].append(line["view_count"])
        data_dict["title"].append(line["title"])
        data_dict["channel_id"].append(line["channel_id"])
        print(count)

    df = pd.DataFrame(data = data_dict)
    df.to_csv("youniverse_dataset.csv", chunksize=1000000)
            
