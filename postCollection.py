import requests
import json
import html


headers = {"User-Agent": "my_app/1.0"}

# Get posts from r/stocks (limit to 5)
url = "https://www.reddit.com/r/stocks/hot.json?limit=5"
response = requests.get(url, headers=headers)
posts = response.json()["data"]["children"]

# Loop through posts and fetch their comments (first 5 posts)
def extract_comments(children):
    """Recursively extract comments and their replies from a Reddit comments 'children' list.

    Each returned comment dict contains: author, body, score, and optional 'replies' list.
    """
    extracted = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        d = child.get("data", {})
        body = d.get("body")
        if body is None:
            continue
        # Unescape HTML entities so things like "&amp;" become "&"
        body = html.unescape(body)
        comment_obj = {"author": d.get("author"), "body": body, "score": d.get("score")}

        # 'replies' may be an empty string or a dict containing another 'data'->'children'
        replies = d.get("replies")
        if isinstance(replies, dict):
            try:
                reply_children = replies.get("data", {}).get("children", [])
                reply_list = extract_comments(reply_children)
                if reply_list:
                    comment_obj["replies"] = reply_list
            except Exception:
                # best-effort: skip nested replies on error
                pass

        extracted.append(comment_obj)
    return extracted

posts_cache = []
for post in posts[:5]:  # first 5 posts
    post_id = post["data"]["id"]
    post_title = html.unescape(post["data"]["title"])
    print(f"\nPost: {post_title}")

    comments_url = f"https://www.reddit.com/r/stocks/comments/{post_id}.json"
    comments_response = requests.get(comments_url, headers=headers)

    # Safely extract comments list
    comments_data = []
    if comments_response.ok:
        try:
            comments_json = comments_response.json()
            # comments are in the second element
            comments_data = comments_json[1]["data"].get("children", [])
        except Exception:
            comments_data = []

    # Build post entry and extract nested comments (replies)
    post_entry = {"id": post_id, "title": post_title, "comments": extract_comments(comments_data)}
    posts_cache.append(post_entry)

# Write collected posts/comments to 'postcache' as JSON
try:
    with open('postCache.JSON', 'w', encoding='utf-8') as f:
        json.dump(posts_cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(posts_cache)} posts to 'postcache'")
except Exception as e:
    print("Failed to write postcache:", e)
        