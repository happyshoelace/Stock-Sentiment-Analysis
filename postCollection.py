import requests
import json
import html
import re
import os


headers = {"User-Agent": "my_app/1.0"}


def clean_text(value):
    """Remove newline characters and normalize repeated whitespace."""
    if value is None:
        return ""
    value = html.unescape(value)
    value = value.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def fetch_json(url, headers, params=None, timeout=20):
    """Fetch JSON safely and return None when the response is not valid JSON."""
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

    if not response.ok:
        print(f"Request failed for {url}: HTTP {response.status_code}")
        return None

    try:
        return response.json()
    except ValueError:
        preview = (response.text or "")[:200].replace("\n", " ")
        print(f"Non-JSON response for {url}: {preview}")
        return None

# Get posts from a subreddit
subreddit = "wallstreetbets"
url = f"https://www.reddit.com/r/{subreddit}/hot.json"
posts_payload = fetch_json(url, headers=headers, params={"limit": 100})
posts = []
if posts_payload:
    posts = posts_payload.get("data", {}).get("children", [])

if not posts:
    print("No posts retrieved. Exiting without writing updates.")

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
        body = clean_text(body)
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
for post in posts:  # first 5 posts
    post_id = post["data"]["id"]
    post_title = clean_text(post["data"]["title"])
    print(f"\nPost: {post_title}")

    comments_url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    comments_json = fetch_json(comments_url, headers=headers)

    # Safely extract comments list
    comments_data = []
    if isinstance(comments_json, list):
        try:
            # comments are in the second element
            comments_data = comments_json[1]["data"].get("children", [])
        except Exception:
            comments_data = []

    # Build post entry and extract nested comments (replies)
    post_entry = {"id": post_id, "title": post_title, "comments": extract_comments(comments_data)}
    posts_cache.append(post_entry)

# Append collected posts/comments to existing cache JSON (if present)
try:
    existing_posts = []
    if os.path.exists('postCache.JSON'):
        with open('postCache.JSON', 'r', encoding='utf-8') as f:
            try:
                existing_posts = json.load(f)
                if not isinstance(existing_posts, list):
                    existing_posts = []
            except json.JSONDecodeError:
                existing_posts = []

    combined_posts = existing_posts + posts_cache

    with open('postCache.JSON', 'w', encoding='utf-8') as f:
        json.dump(combined_posts, f, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(posts_cache)} new posts. "
        f"Cache now contains {len(combined_posts)} posts."
    )
except Exception as e:
    print("Failed to write postcache:", e)
        