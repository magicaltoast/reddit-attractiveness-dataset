from psaw import PushshiftAPI
from datetime import datetime
from queue import Queue
from tqdm.auto import tqdm
from threading import Thread
from praw import Reddit
import re
import requests
import numpy as np
import cv2
import numpy as np
import json
from pathlib import Path
from config import REDDIT_AUTH

img_ext = set([".png", ".jpg", ".jpeg"])

r = Reddit(
    **REDDIT_AUTH
)
api = PushshiftAPI(r)

DONE = object()


def worker(queue, subreddit, limit, time_cache):
    time_cache.mkdir(exist_ok=True)
    file_path = time_cache / subreddit

    if file_path.exists():
        with open(file_path) as f:
            lines = f.readlines()
        before = int(lines[-2].strip())
        print(before)
    else:
        before = int(datetime.now().timestamp())

    with open(file_path, "a") as f:
        while True:
            for _ in range(10):
                try:
                    posts = api.search_submissions(
                        before=before, subreddit=subreddit, limit=limit
                    )
                    break
                except Exception as E:
                    print(E)
                    continue

            first = next(posts, None)

            if not first:
                queue.put(DONE)
                return

            queue.put(first)
            f.write(str(int(first.created_utc)) + "\n")
            f.flush()

            for post in posts:
                queue.put(post)
                last = post

            before = int(last.created_utc)


def iter_posts(subreddit, queue_size, time_cache=Path("time-cache/")):
    q = Queue(queue_size)
    t = Thread(
        target=worker,
        args=[q, subreddit, 500, time_cache],
    )
    t.start()
    p = tqdm(desc=subreddit)

    while (item := q.get()) is not DONE:
        yield item
        p.update()
        p.set_description(f"{subreddit} {datetime.fromtimestamp(item.created_utc)}")


def load_data(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def save_data(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder)


def download_img_from_url(url):
    r = requests.get(url, stream=True).raw
    img = np.asarray(bytearray(r.read()), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def download_single_img(post):
    if Path(post.url).suffix not in img_ext:
        return
    return download_img_from_url(post.url)


def download_multiple_images(post):
    image = download_single_img(post)

    if image is not None:
        return [(image, post.url)]

    images = []
    for media in getattr(post, "media_metadata", {}).values():
        url = media["p"][-1]["u"]
        images.append((download_img_from_url(url), url))

    return images
