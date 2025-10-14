# 📚 RAG Assistant — Automated Research Summarizer for LLM Papers
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![arXiv Powered](https://img.shields.io/badge/Data-arXiv-orange)](https://arxiv.org)


RAG Assistant is an end-to-end **automated Retrieval-Augmented Generation (RAG)** system that continuously fetches the newest **LLM-related research papers** from **arXiv**, processes them on the fly, and enables **interactive question-answering and daily summarization** of emerging AI research trends.  

This project demonstrates how to combine **retrieval**, **generation**, and **automation** to build a real-world research assistant — fully open-source and cloud-ready.

---
## 📑 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Automation](#-automation)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [About This Project](#-about-this-project)

---

## 🚀 Features

- **Automated Paper Ingestion** – Fetches the latest LLM-related research from arXiv using its open API (no manual downloads).
- **Streaming PDF Processing** – Extracts and chunks text from PDFs in real time using `pypdf` and `requests`.
- **Vector Embeddings** – Converts text into dense semantic vectors via `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
- **Semantic Retrieval** – Retrieves the most relevant paper segments via FAISS or Pinecone vector stores.
- **LLM Response Generation** – Combines retrieved research context with an LLM (OpenAI or Hugging Face) for accurate answers and summaries.
- **Automated Daily Digest** – Generates a concise markdown summary (“LLM Research Digest”) of new papers each day.
- **Interactive UI** – Streamlit web app for live research Q&A and visualization.
- **Cloud-First Workflow** – Streams data from arXiv (no local raw file storage needed).

---

## 🧰 Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| Programming | Python 3.10+ |
| Data Source | `arxiv` API |
| PDF Parsing | `pypdf`, `requests` |
| NLP Models | `sentence-transformers`, `transformers` |
| Vector Store | FAISS / Pinecone |
| LLM Backend | OpenAI GPT-4o / Hugging Face Inference API |
| Frontend | `streamlit` |
| Automation | `cron` / GitHub Actions (daily refresh) |
| Optional | `LangChain`, `LlamaIndex`, `vLLM`, `Weights & Biases` |

---

## 🧩 Project Structure

```
rag-assistant/
├── src/
│ ├── fetch_arxiv.py # Search & fetch latest LLM papers from arXiv
│ ├── ingest_stream.py # Stream, parse, and chunk PDF text
│ ├── embed_index.py # Build FAISS index & store metadata
│ ├── rag_answer.py # Query, retrieve, and generate LLM-based answers
│ ├── scheduler_daily.py # Automate daily index refresh + digest summary
│ └── utils/clean.py # Optional text cleanup helpers
├── indexes/ # FAISS index + metadata JSON
├── reports/ # Daily LLM Digest markdown files
├── app.py # Streamlit interface for live QA
├── requirements.txt
├── .env # API keys and configuration
└── README.md
```


---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/rag-assistant.git
cd rag-assistant

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # on Mac/Linux

# Install dependencies
pip install -r requirements.txt

### Environment Variables (.env)
OPENAI_API_KEY=sk-xxxx
HF_TOKEN=hf_xxx
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
---


## 🧠 Quick Start

1. Fetch Latest Papers from arXiv
```bash
python src/fetch_arxiv.py
```
2. Stream, Parse & Chunk PDFs
```bash
python src/ingest_stream.py
```

3. Build Vector Index
```bash
python src/embed_index.py
```

4. Ask Questions
```bash
python src/rag_answer.py
```

or run the interactive Streamlit app:
```bash
streamlit run app.py
```

Then visit 👉 [http://localhost:8501](http://localhost:8501) to explore the Streamlit research assistant UI.

## ⚡ Automation

You can automate daily updates and summaries using:

GitHub Actions (recommended):
Run python src/scheduler_daily.py on a daily cron to:

- Fetch new papers

- Rebuild embeddings

- Generate a new LLM Research Digest in /reports/


### Example GitHub Action (cron.yaml)
```yaml
name: Daily Digest
on:
  schedule:
    - cron: "0 14 * * *"  # every day at 14:00 UTC
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python src/scheduler_daily.py
```


## 📈 Roadmap

- Add reranking model for improved retrieval precision

- Implement hybrid (keyword + vector) retrieval

- Support multi-source ingestion (CORE, Hugging Face datasets)

- Enhance summarization metrics (faithfulness, recall)

- Deploy Daily Digest to Slack / Discord

- vLLM or Triton serving for low-latency LLM responses


## 🪪 License

This project is licensed under the MIT License.

## 🌟 Acknowledgements

- [Sentence-Transformers](https://www.sbert.net)
- [LangChain](https://www.langchain.com)
- [Hugging Face Transformers](https://huggingface.co)
- [OpenAI API](https://platform.openai.com)
- [arXiv.org](https://arxiv.org)

---

## 💬 About This Project

This project was developed as part of a **self-learning journey** to deepen my understanding of
**Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** systems.

With ChatGPT as a learning companion, I designed and implemented this project from scratch, covering data ingestion, vector search, model prompting, and automated summarization pipelines.
The goal is to bridge the gap between theory and real-world LLM deployment while building
a reusable framework for exploring GenAI research applications.

> 🧭 *This project is part of my ongoing exploration in AI/ML system design and Generative AI engineering.*

---
