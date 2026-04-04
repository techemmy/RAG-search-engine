import os
import json
from typing import List

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def get_movies():
    with open("data/movies.json", "r") as f:
        data = json.load(f)
    return sorted(data["movies"], key=lambda x: x["id"])


def read_stopwords() -> List[str]:
    with open("data/stopwords.txt", "r") as f:
        stopwords = set(line.strip() for line in f.read().splitlines())
    return list(stopwords)
