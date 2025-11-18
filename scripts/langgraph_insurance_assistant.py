import os
import json
from dotenv import load_dotenv
from collections import deque
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
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
INSURANCE_CATEGORIES = ["policy_types", "benefits", "eligibility", "claims"]

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

# Greeting Agent
def greeting_agent(query, history_context):
    system = ("system", "You are a friendly life insurance assistant. Respond politely to greetings or small talk about insurance.")
    user = ("user", query)
    return query_llm([system, user])

# Unrelated Agent
def unrelated_agent(*args):
    return "Sorry, I can only answer questions about life insurance policies, benefits, eligibility, and claims."

# Category Agent Template
def category_agent(category, query, history_context):
    data = category_data.get(category, [])
    content = "\n".join([f"{d['title']}: {d['content']}" for d in data])
    history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history_context])

    system = ("system", f"You answer strictly using {category} data.")
    user = ("user", f"Content:\n{content}\n\nHistory:\n{history_text}\nQuery:\n{query}")
    return query_llm([system, user])

# Aggregator Agent
def aggregate_answers(query, answers):
    combined = "\n".join([f"{k.upper()}:\n{v}\n" for k, v in answers.items()])
    system = ("system", "Combine and refine these category answers into one clean response.")
    user = ("user", f"Query: {query}\nAnswers:\n{combined}")
    return query_llm([system, user])

# ============================================================
# LANGGRAPH MULTI-AGENT WORKFLOW
# ============================================================
def build_langgraph(categories):
    graph = StateGraph(dict)

    # Add category nodes
    for cat in categories:
        def make_node(c):
            def node(state):
                state["answers"][c] = category_agent(c, state["query"], state["history"])
                return state
            return node

        graph.add_node(cat, make_node(cat))

    # Merge node
    def merge_node(state):
        state["final"] = aggregate_answers(state["query"], state["answers"])
        return state

    graph.add_node("merge", merge_node)

    # Edges
    first = categories[0]
    graph.set_entry_point(first)

    for c in categories:
        graph.add_edge(c, "merge")

    graph.add_edge("merge", END)
    return graph.compile()

def run_langgraph(categories, query, history_context):
    app = build_langgraph(categories)
    init_state = {"query": query, "history": list(history_context), "answers": {}}
    result = app.invoke(init_state)
    return result["final"]

# ============================================================
# CHAT LOOP
# ============================================================
query_cache = load_cache()
MAX_CACHE_SIZE = 20
def chat():
    print("\nWelcome to Life Insurance Support Assistant! LangGraph + LangChain")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        cached_answer = get_cached_answer(query_cache, q)

        # CACHE HANDLING
        if cached_answer:
            print("[Debug] Categories (cached):", query_cache[q + "_categories"])
            print("Assistant:", query_cache[q])
            continue

        cats = classify_categories(q)
        query_cache[q + "_categories"] = cats
        save_cache(query_cache)

        # Greeting or Unrelated → no LangGraph
        if cats == ["greeting"]:
            response = greeting_agent(q, history)
        elif cats == ["unrelated"]:
            response = unrelated_agent(q, history)

        # Insurance categories → single agent
        elif len([c for c in cats if c in INSURANCE_CATEGORIES]) == 1:
            cat = [c for c in cats if c in INSURANCE_CATEGORIES][0]
            response = category_agent(cat, q, history)

        # Multiple insurance categories → LangGraph
        else:
            multi = [c for c in cats if c in INSURANCE_CATEGORIES]
            response = run_langgraph(multi, q, history)

        mode = (
            "Greeting"
            if cats == ["greeting"] else
            "Unrelated"
            if cats == ["unrelated"] else
            "Single-Agent"
            if len([c for c in cats if c in INSURANCE_CATEGORIES]) == 1 else
            "LangGraph-Multi-Agent"
        )
        print(f"[Debug] Categories: {cats} | Mode={mode}")

        # Save & Print
        history.append((q, response))
        update_cache(query_cache, q, response)
        save_cache(query_cache)
        print("Assistant:", response)

if __name__ == "__main__":
    chat()
