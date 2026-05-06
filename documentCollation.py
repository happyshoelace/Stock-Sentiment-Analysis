import csv
import pprint as pp
import json
import re
from pathlib import Path

SECURITY_NAME_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc", "ltd",
    "llc", "class", "common", "stock", "ordinary", "shares", "holdings", "group"
}


def _tokenize(value):
    return re.findall(r"[a-z0-9]+", (value or "").lower())


def _security_name_is_close_match(text, security_name):
    if security_name.lower() in SECURITY_NAME_STOPWORDS:
        for stopword in SECURITY_NAME_STOPWORDS:
            security_name = security_name.lower().replace(stopword, "")

    if security_name.strip() in text.lower():
        return True

    return False

def findStockMentions(text, stockAbbreviationList, tracker):
    for stock, security_name in stockAbbreviationList.items():
        stock = (stock or "").strip()
        security_name = (security_name or "").strip()

        if not stock or not security_name:
            continue

        stock_pattern = r"\b" + re.escape(stock) + r"\b"

        if re.search(stock_pattern, text, re.IGNORECASE) or _security_name_is_close_match(text, security_name):
            if stock not in tracker:
                tracker[stock] = [text]
            else:
                tracker[stock].append(text)
    return tracker

    
stockDefinitions = {}

with open('stockabbreviations.csv') as  f:
    reader = csv.DictReader(f, delimiter="|")
    for data in reader:
        stockDefinitions[data["Symbol"]] = data["Security Name"]

with open('postCache.JSON', "r", encoding="utf-8") as cache:
    cacheDict = json.load(cache)

# pp.pprint(cacheDict[0]["comments"][0])

stockMentions = {}

for comment in cacheDict[0]["comments"]:
        stockMentions = findStockMentions(comment["body"], stockDefinitions, stockMentions)
        for reply in comment.get("replies", []):
            stockMentions = findStockMentions(reply["body"], stockDefinitions, stockMentions)


pp.pprint(stockMentions)


output_path = Path("stockMentions.json")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(stockMentions, f, ensure_ascii=False, indent=2)

def check(stock, mentions):
    print(f"Stock: {stock}")
    for mention in mentions:
        if stock in mention or _security_name_is_close_match(mention, stockDefinitions[stock]):
            print(f"---\n {mention.replace(stock, f'**{stock}**')}\n---")

  

# with open('stockMentions.json', "r", encoding="utf-8") as f:
#     data = json.load(f)

# for stock, mentions in data.items():
#     check(stock, mentions)




