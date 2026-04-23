import csv
import pprint as pp
import json

def findStockMentions(text, stockAbbreviationList, tracker):
    for stock in stockAbbreviationList.keys():
        if stock in text:
            print(f"I found {stock}!")
    



stockDefinitions = {}

with open('stockabbreviations.csv') as  f:
    reader = csv.DictReader(f, delimiter="|")
    for data in reader:
        stockDefinitions[data["Symbol"]] = data["Security Name"]

with open('postCache.JSON', "r", encoding="utf-8") as cache:
    cacheDict = json.load(cache)

pp.pprint(cacheDict[0]["comments"][0])
findStockMentions(cacheDict[0]["comments"][0]["body"], stockDefinitions, [])