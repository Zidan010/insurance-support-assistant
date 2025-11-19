# Life Insurance Support Assistant

A lightweight **agentic life-insurance assistant** built using **LangChain + Groq**, designed to answer domain-specific questions about **policy types**, **benefits**, **eligibility**, and **claims**, using structured category-based reasoning.

This project fulfills the requirements of the assessment task by implementing:
- A fully working CLI interface
- Use of LLM(GROQ API) for query processing and response along with langchain
- Context-aware conversation history
- JSON-based cache
- Multiple agents running in parallel
- Robust error handling & fallback LLM
- LangGraph integration for multi-agent workflows
- A clean, modular pipeline
- Category-based LLM routing(policy types, benefits, eligibility, and
claims)
- A well-documented dataset and workflow
- Also added a fastapi endpoint for backend api understanding

---
## 🎬 Demo Video
[Watch Demo Video](https://drive.google.com/file/d/1pzCxN4e9jANaau0QwM3LpwHXMUC1kjS0/view?usp=drive_link)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the CLI](#-running-the-cli)
- [Running via FastAPI](#-running-via-fastapi)
- [Agentic Workflow](#-agentic-workflow)
  - [Data Preparation](#data-preparation)
  - [Category Classification](#1️⃣-category-classification)
  - [Routing](#2️⃣-routing)
  - [Category Agents](#3️⃣-category-agents)
  - [Parallel & LangGraph Execution](#4️⃣-parallel--langgraph-execution)
  - [Aggregation Agent](#5️⃣-aggregation-agent)
- [Fallback Model Logic](#-fallback-model-logic)
- [Caching & History](#-caching--history)
- [Testing](#-testing)
- [Evaluation Criteria Mapping](#-evaluation-criteria-mapping)
- [Why Groq Instead of OpenAI](#-why-groq-instead-of-openai)
- [Why Not RAG and No Vector Store / Embeddings?](#-why-not-rag-and-no-vector-store--embeddings)
- [Future Improvements](#-future-improvements)
- [Troubleshooting](#-troubleshooting)

---

## 📝 Overview

This project implements a **life-insurance support assistant** that provides:
- Conversational question answering
- Category-aware reasoning
- Multi-category routing
- Parallel agent execution
- Context handling over multiple turns, also cache management for limiting token usage.

All knowledge is stored in curated JSON datasets, making the system deterministic, explainable, and easy to audit.

---

## ✨ Features

### ✔️ Core Features

- Life-insurance Q&A across:
  - **Policy Types**
  - **Benefits**
  - **Eligibility**
  - **Claims**
- Small-talk support:
  - Classifier detects **greetings** and replies naturally.
- Out-of-domain handling:
  - Answers politely when queries fall outside the insurance domain.
- Classification agent chooses:
  - One category
  - Multiple categories
  - greeting
  - unrelated
- Parallel execution using **ThreadPoolExecutor**.
- Aggregation agent combines multi-category answers into one refined response.
- Caching for repeated queries with JSON storage (last 20 insurance queries).
- Conversation history (last 5 turns) is included for context in responses.

### ✔️ Reliability & Robustness

- Base + fallback LLM handled through LangChain.
- Errors gracefully caught and returned with user-friendly messages.
- Debug lines show:
  - Detected categories
  - Agents executed

---

## 📂 Project Structure
```
life_insurance_agent/
├─ .env
├─ README.md
├─ requirements.txt
├─scripts
│  ├─ clean_and_analyze.py
│  ├─ preprocess_and_structure.py
│  ├─ insurance_assistant.py
│  ├─ insurance_assistant_langgraph.py
│  └─  cache_utils.py
├─ database/
│  ├─ query_cache.json
│  └─ raw
│     ├─ policy_types.json
│     ├─ benefits.json
│     ├─ eligibility.json
│     └─  claims.json
│  ├─ cleaned/
│  └─ final/
│     ├─ policy_types/policy_types_data.json
│     ├─ benefits/benefits_data.json
│     ├─ eligibility/eligibility_data.json
│     ├─ claims/claims_data.json
│     └─ category_classification.json
└─ demo_test_queries.txt
```

---

## 🛠 Prerequisites

- Python **3.10+**
- A valid **Groq API Key**
- Virtual environment recommended

---

## ⚙️ Installation & Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd life_insurance_agent
```

### 2. Create a virtual environment
```bash
python -m venv _env
```

**Activate:**

- **Windows:**
```bash
  _env\Scripts\activate
```
- **macOS / Linux:**
```bash
  source _env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure .env
Add your Groq API key:
```ini
GROQ_API_KEY=sk-xxxxxxxx
```

### 5. Ensure dataset is prepared
```bash
python clean_and_analyze.py
python preprocess_and_structure.py
```

---

## ▶️ Running the CLI

**Standard LangChain + Groq mode:** 
```bash
python insurance_assistant.py
```

**LangGraph (only implemented in multi-agent mode):**
```bash
python langgraph_insurance_assistant.py
```

**Example:**
```
You: What are the types of life insurance policies?
[Debug] Selected categories: ['policy_types']
Assistant: The types of life insurance policies include term insurance, whole life insurance, endowment plans, and universal life policies....
```
## ▶️ Running via FastAPI

The assistant can also be accessed through a REST API using FastAPI.

### 1. Start the API server
```bash
python fastapi_insurance_assistant.py
```
---

## 🧠 Agentic Workflow

### Data Preparation

- Raw insurance data collected from official sources
- Cleaned and structured into JSON
- Each category stored separately for deterministic access

### 1️⃣ Category Classification

- Classifier reads `category_classification.json`
- Returns:
  - `["policy_types"]`
  - `["policy_types", "claims"]`
  - `["greeting"]`
  - `["unrelated"]`

### 2️⃣ Routing

- `greeting` → greeting agent
- `unrelated` → polite refusal
- Single insurance category → single agent
- Multiple insurance categories → parallel agents → aggregated (LangGraph used here optionally)

### 3️⃣ Category Agents

Each category has its own function:
- `policy_types_agent`
- `benefits_agent`
- `eligibility_agent`
- `claims_agent`

Each agent receives:
- User query
- Last 5 conversation turns
- Only the dataset(knowledge) belonging to its category

### 4️⃣ Parallel & LangGraph Execution

- Parallel execution using **ThreadPoolExecutor** for multi-category queries
- **LangGraph** used to orchestrate multiple insurance agents and aggregate answers
- Provides structured, reproducible workflow for multi-agent responses

### 5️⃣ Aggregation Agent

- Merges multiple category answers into one coherent response
- Ensures context-aware, non-contradictory answer

---

## 🛡 Fallback Model Logic

**Base model:** `llama-3.3-70b-versatile`

**Fallback model:** `llama-3.3-7b-instant`

**Logic:**
1. Try base model
2. If timeout/error → fallback
3. If fallback fails → user-friendly final message

---

## 🔁 Caching & History

- Cache stored in JSON (`query_cache.json`) and persists across sessions
- Only last 20 queries are saved
- Conversation history (last 5 turns) included for context
- Cache updated dynamically when new queries are answered

---

## 🧪 Testing

**Automated test set:** `demo_test_queries.txt`
- 10 single-category questions
- 10 multi-category questions
- 5 small-talk / out-of-domain questions

---

## 📊 Evaluation Criteria Mapping

### ✔️ Functionality

- Domain-accurate responses
- Handles greetings, unrelated, and multi-category insurance queries

### ✔️ Architecture

- Modular, clean design
- Agent-based pipeline
- Parallel execution & LangGraph for multi-agent orchestration
- Base + fallback LLM

### ✔️ Documentation

- Clear README & inline comments
- Dataset preparation pipeline

### ✔️ User Experience

- Natural conversation flow
- Debug category tracing
- Context-aware responses
- Robust fallbacks

---

## ⚡ Why Groq Instead of OpenAI?

- Faster inference with Groq LPU acceleration
- Cheaper for frequent classification + agent calls, as i didn't have openai api
- Native LangChain support via `ChatGroq`
- Ideal for multi-agent, parallel workflows

---

## 🚫 Why Not RAG and No Vector Store / Embeddings?

- Dataset is small, structured, and categorical
- Direct category lookup is deterministic & auditable
- Embeddings unnecessary for tightly-scoped domain
- Ensures consistent evaluation

---

## 🚀 Future Improvements

- Real RAG system with embeddings + vector store for large datasets
- Web API + UI (FastAPI + React/Next.js)
- Logging, analytics, and model performance tracking
- Input safety & structured refusal handling

---

## 🐞 Troubleshooting

- **Model failed:** Check API key, connection, rate limits
- **No categories detected:** Improve classification prompt
---

