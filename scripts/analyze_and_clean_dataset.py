import json
import os
from collections import Counter

# Folder path of the raw JSON files 
DATA_FOLDER = "../database/raw"

# Categories that we are covering in the dataset
categories = ["policy_types", "benefits", "eligibility", "claims"]

def load_data(category):
    file_path = os.path.join(DATA_FOLDER, f"{category}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def clean_data(data):
    cleaned = []
    seen_ids = set()
    for entry in data:
        # Ensuring all required fields exist in our dataset
        if all(key in entry for key in ["id", "title", "content", "source_name", "source_url"]):
            if entry["id"] not in seen_ids:
                cleaned.append(entry)
                seen_ids.add(entry["id"])
    return cleaned

def analyze_data(all_data):
    print("=== Dataset Analysis ===")
    total_entries = 0
    for category, data in all_data.items():
        print(f"Category '{category}': {len(data)} entries")
        total_entries += len(data)
    print(f"Total entries across all categories: {total_entries}")

    # just checking Top 10 words in titles
    all_titles = []
    for data in all_data.values():
        all_titles.extend([entry["title"] for entry in data])
    words = " ".join(all_titles).lower().split()
    word_counts = Counter(words)
    print("\nTop 10 frequent words in titles:")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")

def main():
    all_data = {}
    for category in categories:
        data = load_data(category)
        cleaned = clean_data(data)
        all_data[category] = cleaned

        # Saving cleaned data in separate path
        cleaned_folder = os.path.join("../database/cleaned")
        os.makedirs(cleaned_folder, exist_ok=True)
        with open(os.path.join(cleaned_folder, f"{category}_cleaned.json"), "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

    analyze_data(all_data)

if __name__ == "__main__":
    main()
