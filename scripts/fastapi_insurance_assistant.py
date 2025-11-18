import os
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_groq import ChatGroq
from cache_utils import load_cache, save_cache, get_cached_answer, update_cache

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in .env")

# Paths and Constants
FINAL_FOLDER = "../database/final"
CLASSIFICATION_FILE = os.path.join(FINAL_FOLDER, "category_classification.json")
CATEGORIES = ["greeting", "unrelated", "policy_types", "benefits", "eligibility", "claims"]
MAX_HISTORY = 5
MAX_CACHE_SIZE = 20

# Load Classification Data
try:
    with open(CLASSIFICATION_FILE, "r", encoding="utf-8") as f:
        classification_data = json.load(f)
except Exception as e:
    print(f"Error loading classification data: {e}")
    classification_data = []

# Load Category Data
category_data = {}
for cat in ["policy_types", "benefits", "eligibility", "claims"]:
    path = os.path.join(FINAL_FOLDER, cat, f"{cat}_data.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            category_data[cat] = json.load(f)
    except Exception as e:
        print(f"Error loading category data '{cat}': {e}")
        category_data[cat] = []

# Conversation History
history = deque(maxlen=MAX_HISTORY)

# LangChain Groq Models
BASE_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.3-7b-instant"

llm_base = ChatGroq(model=BASE_MODEL, temperature=0.3, max_retries=2)
llm_fallback = ChatGroq(model=FALLBACK_MODEL, temperature=0.3, max_retries=2)

# LLM Query Function
def query_llm(messages):
    try:
        resp = llm_base.invoke(messages)
        return resp.content
    except Exception as e:
        print(f"Error with base model {BASE_MODEL}: {e}")
        try:
            resp2 = llm_fallback.invoke(messages)
            return resp2.content
        except Exception as e2:
            print(f"Fallback model {FALLBACK_MODEL} failed: {e2}")
            return "Sorry, I am unable to process your request at the moment. Please try again later."


# Classification Agent
def classify_categories(user_query):
    context_text = "\n".join(
        [f"{d['category_name']}: {d['description']}" for d in classification_data]
    )
    system_msg = ("system",
        """
        You are a Life Insurance Query Classifier.
        - If query is about insurance → return relevant categories.
        - If greeting → include 'greeting'.
        - If unrelated → include 'unrelated'.
        Return ONLY a Python list. Example: ["policy_types", "claims"]
        """
    )
    user_msg = ("user", f"Query: {user_query}\nContext: {context_text}")
    response = query_llm([system_msg, user_msg]).strip()

    import ast
    try:
        cats = ast.literal_eval(response)
        return [c for c in cats if c in CATEGORIES] or ["unrelated"]
    except:
        return ["unrelated"]

# Greeting & Unrelated Agents
def greeting_agent(user_query, history_context):
    system = ("system", "You are a friendly life insurance assistant. Respond politely to greetings or small talk about insurance.")
    user = ("user", user_query)
    return query_llm([system, user])

def unrelated_agent(user_query, history_context):
    return "Sorry, I can only answer questions about life insurance policies, benefits, eligibility, and claims."

# Category Agents
def category_agent_template(category, user_query, history_context):
    data = category_data.get(category, [])
    content_text = "\n".join([f"{d['title']}: {d['content']}" for d in data])
    history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history_context])
    system_msg = ("system", f"You answer strictly using {category} data.")
    user_msg = ("user", f"Content:\n{content_text}\n\nHistory:\n{history_text}\nQuery:\n{query}")
    return query_llm([system_msg, user_msg])

policy_types_agent = lambda u, h: category_agent_template("policy_types", u, h)
benefits_agent = lambda u, h: category_agent_template("benefits", u, h)
eligibility_agent = lambda u, h: category_agent_template("eligibility", u, h)
claims_agent = lambda u, h: category_agent_template("claims", u, h)

category_agents_map = {
    "policy_types": policy_types_agent,
    "benefits": benefits_agent,
    "eligibility": eligibility_agent,
    "claims": claims_agent,
    "greeting": greeting_agent,
    "unrelated": unrelated_agent
}

# Aggregation Agent
def aggregate_answers(user_query, answers):
    combined_text = "\n".join([f"{cat}: {ans}" for cat, ans in answers.items()])
    system_msg = ("system", "Combine and refine these category answers into one clean response.")
    user_msg = ("user", f"Query: {user_query}\nAnswers:\n{combined_text}")
    return query_llm([system_msg, user_msg])

# Load JSON cache
query_cache = load_cache()

# FastAPI Setup
app = FastAPI(title="Life Insurance Support Assistant API")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_query = request.query.strip()
    cached_answer = get_cached_answer(query_cache, user_query)
    if cached_answer:
        response = cached_answer
    else:
        # Classify
        categories = classify_categories(user_query)
        query_cache[user_query+'_categories'] = categories
        save_cache(query_cache)

        answers = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(category_agents_map[cat], user_query, history): cat for cat in categories}
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    answers[cat] = future.result()
                except Exception as e:
                    answers[cat] = f"Error in {cat} agent: {e}"

        insurance_cats = [c for c in categories if c not in ["greeting", "unrelated"]]
        if len(insurance_cats) > 1:
            insurance_answers = {k: v for k, v in answers.items() if k in insurance_cats}
            response = aggregate_answers(user_query, insurance_answers)
        else:
            response = list(answers.values())[0]

        update_cache(query_cache, user_query, response)
        save_cache(query_cache)

    history.append((user_query, response))
    return {"query": user_query, "response": response, "categories": query_cache.get(user_query+'_categories', [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
