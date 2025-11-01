# chatbot_ollama
An intelligent, conversational chatbot built using **LangChain**, **Streamlit**, and **FAISS**, powered by **Ollama** — a local LLM runtime for open-source models. 

This project demonstrates how to combine retrieval-augmented generation (RAG) and custom prompt workflows to build a fully local, privacy-friendly AI assistant.

---

## 🚀 Features

- 💬 **Conversational Chatbot** — Natural, context-aware responses from open-source LLMs.  
- 🧩 **Retrieval-Augmented Generation (RAG)** — Enhances LLM performance using FAISS-based vector search over local knowledge bases.  
- ⚡ **LangChain Integration** — Modular prompt templates, chaining, and retrieval pipeline.  
- 🌐 **Streamlit Frontend** — Simple and interactive web UI for chatting and visualizing responses.  
- 🛡️ **Local Inference** — Runs entirely on your system with Ollama; no external API keys required.  
- 🔍 **Embeddings Support** — Custom document embeddings for knowledge-based question answering.  

---

## 🏗️ Tech Stack

- **Programming Language:** Python 3.11+  
- **Frameworks & Libraries:**  
  - [LangChain](https://www.langchain.com/) – chaining and RAG pipeline  
  - [Streamlit](https://streamlit.io/) – UI interface  
  - [FAISS](https://faiss.ai/) – vector similarity search  
  - [Ollama](https://ollama.ai/) – open-source LLM runtime  
  - [SentenceTransformers](https://www.sbert.net/) – embedding generation  

---

## ⚙️ Installation
### 1️⃣ Clone this repository
```bash
git clone https://github.com/murugan-bala/chatbot_ollama.git
cd chatbot_ollama

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate       # (Windows) venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt


4️⃣ Install and start Ollama
Download from ollama.ai/download
Pull a model (for example, Mistral):
ollama pull mistral


▶️ Run the App
streamlit run local_bot.py
Then open your browser at http://localhost:8501

🔒 Privacy
Runs completely offline with Ollama — no data leaves your device.

