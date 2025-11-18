import json
import os
from collections import OrderedDict

CACHE_FILE = r"..\database\query_cache.json"
MAX_CACHE_SIZE = 20


def load_cache():
    """Load cache from JSON. If not exists → return empty dict."""
    if not os.path.exists(CACHE_FILE):
        return OrderedDict()

    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            return OrderedDict(data)
    except:
        return OrderedDict()


def save_cache(cache: OrderedDict):
    """Save cache back to JSON."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_cached_answer(cache, query: str):
    """Return answer if exists."""
    return cache.get(query)


def update_cache(cache, query: str, answer: str):
    """Insert new and maintain only the last 10 entries."""
    cache[query] = answer

    # Keep only last MAX_CACHE_SIZE
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)  # remove oldest

    save_cache(cache)
