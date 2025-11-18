import json
import os

# dataset paths
CLEANED_FOLDER = "../database/cleaned"
FINAL_FOLDER = "../database/final"
os.makedirs(FINAL_FOLDER, exist_ok=True)

# Categories and short descriptions for user/query intent understanding by classification agent
category_info = {
    "policy_types": "Covers different life insurance policy types like term life, whole life, endowment, and child plans.",
    "benefits": "Details about the advantages and benefits offered by life insurance policies.",
    "eligibility": "Eligibility criteria including age, health, income, residency, and policy-specific requirements.",
    "claims": "Guidelines on how to file claims, required documents, settlement processes, and common issues."
}

def load_cleaned_data(category):
    file_path = os.path.join(CLEANED_FOLDER, f"{category}_cleaned.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_category_data(category, data):
    category_folder = os.path.join(FINAL_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    # Saving category agent data
    with open(os.path.join(category_folder, f"{category}_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_classification_agent_data(category_info):
    classification_file = os.path.join(FINAL_FOLDER, "category_classification.json")
    classification_data = []
    for cat, desc in category_info.items():
        classification_data.append({
            "category_name": cat,
            "description": desc
        })
    with open(classification_file, "w", encoding="utf-8") as f:
        json.dump(classification_data, f, indent=2, ensure_ascii=False)

def main():
    # Process each category
    for category in category_info.keys():
        data = load_cleaned_data(category)
        save_category_data(category, data)

    # Save classification agent data
    save_classification_agent_data(category_info)
    print("Preprocessing and structuring completed. Data ready for LLM agents.")

if __name__ == "__main__":
    main()
