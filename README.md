# Local RAG Agent ⚙️

Local RAG Agent is a Python project that uses **retrieval-augmented generation (RAG)** over a local dataset of reviews to answer user queries with context. Rather than relying on a remote API or global web search, it uses a local embedding store + language model to produce informed, context-aware responses.

---

## 🚀 Features

- Ingests a CSV file of review data (`realistic_restaurant_reviews.csv`)  
- Embeds review texts into a vector store (via `vector.py`)  
- Supports retrieving top-k similar review snippets for a user query  
- Combines retrieved context + user prompt to generate answer  
- Fully local (no reliance on external search)  
- Simple interface via `main.py` for interactive Q&A  

---

## 🧱 Repository Structure

local-rag-agent/
├── README.md
├── LICENSE
├── realistic_restaurant_reviews.csv
├── requirements.txt
├── vector.py
└── main.py


- **vector.py** — Embedding logic, vector store construction, retrieval functions  
- **main.py** — Entry point for running the agent: takes user input, retrieves context, calls LLM, returns output  
- **realistic_restaurant_reviews.csv** — The dataset of reviews used as knowledge base  
- **requirements.txt** — Python dependencies  

---

## 🛠️ Setup & Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/prarthana-majalikar/local-rag-agent.git
   cd local-rag-agent
   
2. **Create a virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate    # macOS / Linux
    # .venv/Scripts/activate     # OR on Windows:


3. **Install dependencies**
     ```bash
     pip install -r requirements.txt

4. **Setup the local LLM model and Embeddings Model**
   Ollama is an open-source tool that allows you to easily run and manage a wide variety of large language models (LLMs) and other AI models directly on your local machine.
   Download Ollama from  : https://ollama.com/
   Pull the required LLM(https://ollama.com/library/llama3.2) and embeddings model(https://ollama.com/library/mxbai-embed-large)
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large

## 🧪 Example

Here’s an example session (simplified) when running main.py:
   > Enter your question: “Which restaurant has the best food quality?”  
> Retrieved 3 relevant reviews:  
    • “The food was exceptional …”  
    • “I loved the freshness of the ingredients …”  
    • “Quality was okay but overpriced …”  
> Agent answer: “Based on reviews, the highest food quality is attributed to Restaurant A, which is described as ‘exceptional’ and ‘fresh ingredients’ across multiple reviews.”  


## 🔐 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgements

Inspired by TechWithTim’s LocalAIAgentWithRAG project

Uses common LLM / vector search patterns in open source
   
