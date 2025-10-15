# 🌍 Climate Policy Q&A Bot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers complex **climate policy and sustainability questions** using authoritative global documents such as:

- The **Paris Agreement (UNFCCC)**
- **IPCC Synthesis Reports**
- **UNFCCC CMA reports**
- **EU Fit-for-55 Policy Briefings**
- **UNEP Emissions Gap Report 2024**

This project runs **entirely locally** using **Llama 3 (8B)** through **Ollama**, requiring no API keys or cloud dependencies.

---
*(This is where you should put your `demo.gif` file)*
![Climate Bot Demo](screenshots/demo.gif)
---

## 📖 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Project Files](#-project-files)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🧩 Overview

The **Climate Policy Q&A Bot** enables users to ask natural-language questions about climate policy, international agreements, or sustainability goals, and receive **context-grounded answers** based on official reports.

It’s built using **Retrieval-Augmented Generation (RAG)** — a combination of **semantic search** and **local LLM inference**.

💡 **Use Case Examples:**
- Policy analysts checking updates on national NDCs
- Students researching the Paris Agreement
- NGOs summarizing IPCC findings
- Journalists verifying facts on global climate targets

---

## ⚙️ Features

✅ Answers grounded in real, official climate policy documents
✅ Local model inference via **Ollama + Llama 3 (8B)**
✅ Lightweight Retrieval-Augmented Generation pipeline
✅ Cites sources (with PDF page references) for verifiability
✅ Multiple AI personas: `plain`, `policy`, `journalist`
✅ Interactive chat UI built with **Streamlit**
✅ No external API costs or internet dependence

---

## 🧠 Architecture

The diagram below illustrates the RAG data flow, from document ingestion to answer generation.

```plaintext
      ┌──────────────────────────┐
      │  PDF Climate Reports     │
      └────────────┬─────────────┘
                   │
           (ingest.py: Text Extraction & Chunking)
                   │
                   ▼
      ┌──────────────────────────┐
      │  TF-IDF Vector Index     │
      │  (Scikit-learn)          │
      └────────────┬─────────────┘
                   │
           (app.py: Cosine Similarity Search)
                   │
                   ▼
      ┌──────────────────────────┐
      │  Context + Query → LLM   │
      │  (Llama 3 via Ollama)    │
      └──────────────────────────┘
```

---

## 💻 Tech Stack

| Component         | Technology / Library                                       |
| ----------------- | ---------------------------------------------------------- |
| **Language** | Python 3.9+                                                |
| **LLM Backend** | Llama 3 (8B) via **Ollama** |
| **Retrieval** | TF-IDF with **Scikit-Learn** |
| **Interface** | **Streamlit** |
| **Doc Parsing** | **PyMuPDF** |
| **Environment** | venv / virtualenv                                          |

---

## 🧰 Setup & Installation

#### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/climate-policy-rag-bot.git](https://github.com/YOUR_USERNAME/climate-policy-rag-bot.git)
cd climate-policy-rag-bot
```

#### 2️⃣ Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Install and Run Ollama
Follow the instructions on [ollama.ai](https://ollama.ai/) to install Ollama. Then, pull the Llama 3 model and start the server.

```bash
# Pull the model
ollama pull llama3:8b

# Run the server (keep this terminal open)
ollama serve
```

#### 5️⃣ Configure Environment Variables
Create a `.env` file in the project root and add the following:
```env
OPENAI_API_KEY="ollama"
OPENAI_BASE_URL="http://localhost:11434/v1"
LLM_MODEL="llama3:8b"
```

---

## 🚀 Usage

#### 🖥️ Streamlit Mode (Recommended)
Run the Streamlit application for an interactive chat interface.

```bash
streamlit run app.py
```

#### 🧭 CLI Mode
Alternatively, you can ask a single question directly from the terminal:

```bash
python app.py --question "What temperature goal is stated in the Paris Agreement?" --persona plain
```

---

## 📂 Project Files

| File               | Description                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `app.py`           | The main Streamlit web application that provides the chat interface and runs the RAG pipeline.       |
| `ingest.py`        | Parses PDFs, splits them into chunks, and builds the TF-IDF vector index (`index.pkl`).              |
| `eval.py`          | A script to evaluate the chatbot's performance on a custom set of question-answer pairs.            |
| `requirements.txt` | Lists the Python dependencies for the project.                                                 |
| `docs/`            | A directory to store the source PDF and text documents for the chatbot's knowledge base.                        |

---

## 🔮 Future Enhancements

- **Vector Database**: Replace the pickled index with **ChromaDB** or **FAISS** for more scalable vector storage and retrieval.
- **Reranking**: Integrate a **Cross-Encoder** model to re-rank the search results for improved relevance before sending them to the LLM.
- **Deployment**: Package the application in a **Docker** container and deploy it on **Hugging Face Spaces** or Streamlit Cloud.
- **Evaluation Dashboard**: Build a Streamlit dashboard to visualize the results from `eval.py` and track performance changes.

---

## 👩‍💻 Author

**Nivesara Tirupati**
- 🎓 Data Scientist | NLP & ML | Climate Tech | Policy Innovation
- 🔗 [LinkedIn](YOUR_LINKEDIN_URL)
- 🐙 [GitHub](https://github.com/YOUR_GITHUB_USERNAME)
