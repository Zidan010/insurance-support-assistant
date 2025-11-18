import os
import json
from dotenv import load_dotenv
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
import time
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

# LangChain Groq Models (Base + Fallback)
BASE_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.3-7b-instant"

llm_base = ChatGroq(model=BASE_MODEL, temperature=0.3, max_retries=2)
llm_fallback = ChatGroq(model=FALLBACK_MODEL, temperature=0.3, max_retries=2)

# LLM Query Function with Fallback
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
    try:
        context_text = "\n".join([f"{d['category_name']}: {d['description']}" for d in classification_data])
        system_msg = ("system", f"""
                        You are a Life Insurance Assistant Classifier.
                        Rules:
                        1. If the user asks about life insurance, return a list of relevant categories from: policy_types, benefits, eligibility, claims.
                        2. If the query is a greeting or basic small talk, include 'greeting' in the list.
                        3. If the query is completely unrelated, include 'unrelated' in the list.
                        Always return output as a Python list of categories.
                        """)
        user_msg = ("user", f"User Query: {user_query}\nContext: {context_text}")
        
        response = query_llm([system_msg, user_msg]).strip()

        import ast
        try:
            categories = ast.literal_eval(response)
            categories = [c.strip().lower() for c in categories if c.strip().lower() in CATEGORIES]
            return categories if categories else ["unrelated"]
        except Exception:
            # fallback if LLM returns single category as string
            if response.lower() in CATEGORIES:
                return [response.lower()]
            return ["unrelated"]
    except Exception as e:
        print(f"Error in classification: {e}")
        return ["unrelated"]

# Greeting & Unrelated Handlers
def greeting_agent(user_query, history_context):
    system_msg = ("system", "You are a friendly life insurance assistant. Respond politely to greetings or small talk about insurance.")
    user_msg = ("user", user_query)
    return query_llm([system_msg, user_msg])

def unrelated_agent(user_query, history_context):
    return "Sorry, I can only answer questions about life insurance policies, benefits, eligibility, and claims."

# Category Agents
def category_agent_template(category, user_query, history_context):
    try:
        data = category_data.get(category, [])
        content_text = "\n".join([f"{d['title']}: {d['content']}" for d in data])
        history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history_context])
        system_msg = ("system", f"You are a life insurance assistant. Answer only using {category} content.")
        user_msg = ("user", f"Category Content:\n{content_text}\nHistory:\n{history_text}\nUser Query:\n{user_query}")
        return query_llm([system_msg, user_msg])
    except Exception as e:
        print(f"Error in {category} agent: {e}")
        return f"Unable to answer from {category} category."

def policy_types_agent(u, h): return category_agent_template("policy_types", u, h)
def benefits_agent(u, h): return category_agent_template("benefits", u, h)
def eligibility_agent(u, h): return category_agent_template("eligibility", u, h)
def claims_agent(u, h): return category_agent_template("claims", u, h)

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
    try:
        combined_text = "\n".join([f"Answer from {cat} category:\n{ans}" for cat, ans in answers.items()])
        system_msg = ("system", "You are a life insurance assistant. Refine multiple answers into a coherent response.")
        user_msg = ("user", f"User query: {user_query}\nCategory Answers:\n{combined_text}")
        return query_llm([system_msg, user_msg])
    except Exception as e:
        print(f"Error in aggregation agent: {e}")
        return "Sorry, unable to combine category answers at this time."
        
query_cache = load_cache()
MAX_CACHE_SIZE = 20

# CLI Chat Loop
def chat():
    print("\nWelcome to Life Insurance Support Assistant!")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("Assistant: Goodbye!")
                break

            cached_answer = get_cached_answer(query_cache, user_query)

            if cached_answer:
                response = cached_answer

            else:
                # Classify
                categories = classify_categories(user_query)
                query_cache[user_query+'_categories'] = categories
                save_cache(query_cache)
                print(f"[Debug] Selected categories: {categories}")

                answers = {}
                # Run agents in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    # Submit each agent function as a separate task, that returns a Future object
                    futures = {executor.submit(category_agents_map[cat], user_query, history): cat for cat in categories}
                    # futures: maps each Future object to its category name

                    # as_completed yields each Future as soon as it finishes
                    for future in as_completed(futures):
                        cat = futures[future] # Get the category name for this Future
                        try:
                            # Get the result/agent's answer from the Future
                            answers[cat] = future.result()
                        except Exception as e:
                            answers[cat] = f"Error in {cat} agent: {e}"

                # Aggregate if multiple categories (excluding greeting/unrelated)
                insurance_cats = [c for c in categories if c not in ["greeting", "unrelated"]]
                if len(insurance_cats) > 1:
                    insurance_answers = {k: v for k, v in answers.items() if k in insurance_cats}
                    response = aggregate_answers(user_query, insurance_answers)
                else:
                    response = list(answers.values())[0]

                update_cache(query_cache, user_query, response)
                save_cache(query_cache)

            print(f"Assistant: {response}")
            history.append((user_query, response))

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"Error in chat loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    chat()
