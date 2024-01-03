import yaml
import csv
import os
from tqdm import tqdm
import requests
import time 
import pandas as pd
import numpy

def process_playlist(video_snippet):
    temp_dict = {}
    temp_dict["video_id"] = video_snippet["resourceId"]["videoId"]
    temp_dict["title"] = video_snippet["title"]
    temp_dict["video_published_at"] = video_snippet["publishedAt"]
    temp_dict["thumbnail"] = video_snippet["thumbnails"].get("default")
    return temp_dict

def process_video(video_snippet, stats, channel, subCount):
    temp_dict = {}
    temp_dict["title"] = video_snippet["title"]
    temp_dict["video_published_at"] = video_snippet["publishedAt"]
    temp_dict["thumbnail"] = video_snippet["thumbnails"]["default"]["url"]
    if (stats):
        if "viewCount" in stats:
            temp_dict["viewCount"] = stats["viewCount"]
    temp_dict["category"] = video_snippet["categoryId"]
    temp_dict["channel_id"] = channel
    temp_dict["subscriber_count"] = subCount
    return temp_dict

def process_playlist_page(list, count, videoCallList):
    if list != None:
    # process page
        for video in list:
            video_snippet = process_playlist(video["snippet"])
            year = int(video_snippet["video_published_at"][0:4])
            if count <= VIDEOS_PER_CHANNEL and year in TARGET_YEAR:
                videoCallList.append(video_snippet["video_id"])
                count += 1  
            if year < min(TARGET_YEAR):
                count = VIDEOS_PER_CHANNEL + 99                            
            if count > VIDEOS_PER_CHANNEL:
                break

    return count, videoCallList

def blacklist(channel_id):

    with open(BLACKLIST_PATH, "r") as f:
        cur_yaml = yaml.safe_load(f)
        cur_yaml["blacklist"].append(channel_id)
        
    with open(BLACKLIST_PATH, "w") as f:
        print(f"Blacklisting {channel_id}...")
        dump = yaml.safe_dump(cur_yaml)
        f.write(dump)

with open("youtube-video-scraper-master/config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

API_KEY = config["API_KEY"]
CHANNELS_API_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_API_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
VIDEOS_API_URL = "https://www.googleapis.com/youtube/v3/videos"
OUTPUT_FOLDER = "data/dataset_10k_subs/channels"
OUTPUT_FIELDS = ["title", "video_published_at", "thumbnail", "viewCount", "category", "channel_id", "subscriber_count"]
VIDEOS_PER_CHANNEL = 199
MIN_VIDEO_COUNT = 20
TARGET_YEAR = [2022, 2021, 2020, 2019, 2018]
APPROVED_COUNTRIES = ["EN", "AU", "CA", "US", "NZ", "GB", "IR", "ZA", "IN"]
BLACKLIST_PATH = "data/dataset_10k_subs/blacklist.yml"

with open(BLACKLIST_PATH, "r") as f:
    blacklist_channels = yaml.safe_load(f)["blacklist"]

channels_csv = pd.read_csv("data/dataset_10k_subs/channels_10k+_subs.csv")
already_scanned = os.listdir(OUTPUT_FOLDER)

channel_ids = channels_csv.loc[:,"channelId"]
channel_ids = [x for x in channel_ids if x not in blacklist_channels and x + ".csv" not in already_scanned] #remove all instances of blacklisted channels

channel_id_strings = numpy.array_split(channel_ids, len(channel_ids) // 49)
channel_id_strings = [",".join(x) for x in channel_id_strings]

channels_params = {
    "key": API_KEY,
    "part": "contentDetails, statistics, status, snippet",
    "maxResults:": 50,
}

playlist_params = {
    "key": API_KEY,
    "part": "snippet",
    "maxResults": 50,
}

videos_params = {
    "key": API_KEY,
    "part": "snippet, statistics",
    "maxResults": 50,
}

for id_string in channel_id_strings:

    channels_params.update({"id": id_string})

    cR = requests.get(
        CHANNELS_API_URL,
        params=channels_params,
    ).json()

    print(cR)

    if cR.get("items") != None:
        for channel in cR.get("items"):

            channel_id = channel["id"]
            sub_count = channel["statistics"].get("subscriberCount")

            # only parse channel if it has x amount of videos, is public, and is from correct country
            if int(channel["statistics"].get("videoCount")) > MIN_VIDEO_COUNT and channel["status"].get("privacyStatus") == "public":
                if channel["snippet"].get("country") == None or channel["snippet"].get("country") in APPROVED_COUNTRIES:
                    print(f'Channel is from{channel["snippet"].get("country")} and is public')
                    # fetch the playlist ID 
                    uploads_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
                    playlist_params.update({"playlistId": uploads_id})

                    r = requests.get(
                        PLAYLIST_API_URL,
                        params=playlist_params,
                    ).json()

                    if "items" in r:
                        print("There are videos in the channel.")
                        count, videoCallList = process_playlist_page(r.get("items"), 0, [])
                        pageToken = r.get("nextPageToken")
                    
                        while pageToken and count <= VIDEOS_PER_CHANNEL:
                            playlist_params.update({"pageToken": pageToken})
                            r = requests.get(
                                PLAYLIST_API_URL,
                                params=playlist_params,
                            ).json()

                            count, videoCallList = process_playlist_page(r.get("items"), count, videoCallList)
                            pageToken = r.get("nextPageToken")
                            time.sleep(0.01)

                    # split up the video call list into strings with 50 entries
                    if len(videoCallList) // 50 != 0:
                        if (len(videoCallList) / 50).is_integer() == False:
                            truncatedList = videoCallList[-(len(videoCallList)%50):]
                            videoCallList = videoCallList[:-(len(videoCallList)%50)]
                        else:
                            truncatedList = []
                            
                        videoCallListSplit = numpy.array_split(videoCallList, len(videoCallList) // 50)
                        videoCallListSplit = [",".join(x) for x in videoCallListSplit]
                        videoCallListSplit.append(",".join(truncatedList))
                    else:
                        if len(videoCallList) > 0:
                            videoCallListSplit = [",".join(videoCallList)]
                        else:
                            videoCallListSplit = []

                    # reset pageToken for new channel
                    playlist_params.update({"pageToken": None})

                    if len(videoCallListSplit) != 0:

                        videoWriteList = []

                        for videoCallList in videoCallListSplit:
                            if (videoCallList != ""):
                                print(f"WE GOING!:{videoCallList}")
                                videos_params.update({"id": videoCallList})

                                r = requests.get(
                                    VIDEOS_API_URL,
                                    params=videos_params,
                                ).json()
                                print(r)
                                if "items" in r:
                                    pageToken = r.get("nextPageToken")

                                    for video in r["items"]:
                                        videoWriteList.append(process_video(video["snippet"], video["statistics"], channel_id, sub_count))

                                        # process the rest
                                        while pageToken:
                                            videos_params.update({"pageToken": pageToken})
                                            r = requests.get(
                                                VIDEOS_API_URL,
                                                params=videos_params,
                                            ).json()
                                            for video in r["items"]:
                                                videoWriteList.append(process_video(video["snippet"], video["statistics"], channel_id, sub_count))

                                            pageToken = r.get("nextPageToken")
                                            time.sleep(0.01)

                                    with open(
                                        os.path.join(OUTPUT_FOLDER, f"{channel_id}.csv".replace(os.sep, "_")),
                                        "w",
                                        encoding="utf-8",
                                    ) as f:
                                        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
                                        w.writeheader()

                                        print(f"Writing {channel_id}'s data to csv:")
                                        # process first page we already queried
                                        for video in videoWriteList:
                                            w.writerow(video)

                                    # reset pageToken for new channel
                                    videos_params.update({"pageToken": None})
                    else: 
                        blacklist(channel_id)
                else: 
                    blacklist(channel_id)
            else:
                blacklist(channel_id)      

    
    
