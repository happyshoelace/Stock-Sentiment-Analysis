import csv
import pprint
import html
import re
import emoji
from pathlib import Path
import json

stockMentions = {}


def cleanTweet(value):
    """Remove newline characters and normalize repeated whitespace."""
    if value is None:
        return ""
    value = html.unescape(value)
    value = emoji.demojize(value, language="en")
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value

with open('Twitter Dataset/stock_tweets.csv', "r", encoding="utf-8") as  f:
    reader = csv.DictReader(f, delimiter=",")
    for data in reader:
        stockName = data["Stock Name"]
        tweet = cleanTweet(data["Tweet"])
        if stockName not in stockMentions:
            stockMentions[stockName] = [tweet]
        else:
            stockTweetList = stockMentions[stockName]
            stockTweetList.append(tweet)
            stockMentions[stockName] = stockTweetList


pprint.pprint(stockMentions.keys())

output_path = Path("stockMentionsTWITTER.json")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(stockMentions, f, ensure_ascii=False, indent=2)