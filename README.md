# 📚 RAG Assistant — Automated Research Summarizer for LLM Papers
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![arXiv Powered](https://img.shields.io/badge/Data-arXiv-orange)](https://arxiv.org)


RAG Assistant is an end-to-end **automated Retrieval-Augmented Generation (RAG)** system that continuously fetches the newest **LLM-related research papers** from **arXiv**, processes them on the fly, and enables **interactive question-answering and daily summarization** of emerging AI research trends.  

This project demonstrates how to combine **retrieval**, **generation**, and **automation** to build a real-world research assistant — fully open-source and cloud-ready.

---
## 📑 Table of Contents
- [🚀 Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [🧩 Project Structure](#-project-structure)
- [⚙️ Installation](#-installation)
- [🔄 Model Provider Update](#-model-provider-update)
- [🤖 Why GPT-4o-mini](#-why-gpt4omini)
- [🧠 Quick Start](#-quick-start)
- [🧪 Quick Module Tests](#-quick-module-tests)
- [📬 Automated Research Digest](#-automated-research-digest)
- [⚡ Automation](#-automation)
- [📈 Roadmap](#-roadmap)
- [🪪 License](#-license)
- [🌟 Acknowledgements](#-acknowledgements)
- [💬 About This Project](#-about-this-project)

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
| LLM Backend | OpenAI GPT-4o-mini (default) / Hugging Face (facebook/bart-large-cnn only for testing) |
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

## 🔄 Model Provider Update

- During experimentation we found that only `facebook/bart-large-cnn` worked reliably on the free Hugging Face Inference API. Other popular options—Zephyr, Mistral, Gemma—require paid or private hosting and returned 404/403 errors for free-tier tokens.
- To guarantee stable and higher-context summarization, the project now defaults to the OpenAI Chat Completions API with `gpt-4o-mini`.
- Hugging Face support is still in place for lightweight regression testing, but OpenAI is now the recommended provider for day-to-day summaries.

🧩 Note: gpt-5-nano may return empty responses on long-context RAG queries. Use gpt-4o-mini or gpt-4o instead for consistent performance.

## 🤖 Why GPT-4o-mini

OpenAI's gpt-4o-mini is now the default summarizer because it balances cost, speed, and context length:
- **Cheaper** than GPT-4-class models for day-to-day digests.
- **Faster** latency, which keeps scheduled jobs responsive.
- **Good enough** for concise summaries without the overhead of GPT-4o-mini.

For deeper analysis or more nuanced reasoning you can still swap to larger models (e.g., GPT-4o-mini) via the `--provider openai` flag and the `OPENAI_MODEL_ID` environment variable.


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
python src/rag_answer.py --provider openai --model gpt-4o-mini --query "Summarize latest LLM retrieval methods."
```

or run the interactive Streamlit app:
```bash
streamlit run app.py
```

Then visit 👉 [http://localhost:8501](http://localhost:8501) to explore the Streamlit research assistant UI.

## 🧪 Quick Module Tests
Each module can be tested independently before running the full pipeline.


<details>
<summary>Click to expand</summary>

### 1. Fetch Papers
```bash
python src/fetch_arxiv.py --query "large language model" --days-back 3 --max-results 20 --output data/arxiv_results.json
```

### 2. Ingest & Chunk
```bash
python src/ingest_stream.py --input data/arxiv_results.json --limit 20 --output data/chunks.jsonl
# OR
python src/ingest_stream.py --input data/arxiv_results.json --chunk-size 1200 --chunk-overlap 300 --output data/chunks.jsonl

```

### 3. Build & Search Index
```bash
# Build FAISS index
python src/embed_index.py --build --input data/chunks.jsonl --index-path indexes/faiss.index --metadata-path indexes/docs_meta.json

# Search it
python src/embed_index.py --search "What are recent techniques for retrieval-augmented generation?" --k 5

```

### 4. Ask a Question

💡 Environment Variables Note

You can define your API keys in a local .env file so you don’t need to prefix commands each time.
```bash
# Example .env
OPENAI_API_KEY=sk-xxxx
HF_TOKEN=hf_xxx
```

✅ The script will automatically detect the API key from .env.

💡 To temporarily override or use a different key, prefix the command:


```bash
# Using OpenAI GPT-4o-mini (default)
OPENAI_API_KEY=sk-xxxx python src/rag_answer.py --provider openai --model gpt-4o-mini --query "Summarize latest LLM retrieval methods."

# Legacy Hugging Face testing only (facebook/bart-large-cnn)
HF_TOKEN=hf_xxx python src/rag_answer.py --provider hf --query "Summarize latest LLM retrieval methods."

> Note: `--provider hf` is kept only for regression testing with `facebook/bart-large-cnn`. All production runs should use `--provider openai`.
```

### 5. Run Daily Digest
```bash
python src/scheduler_daily.py --days-back 1 --dry-run
```

Full scheduler run (writes reports/YYYY-MM-DD-llm-digest.md): 

```bash
OPENAI_API_KEY=sk-xxxx python src/scheduler_daily.py --days-back 1
```

</details>

## 📬 Automated Research Digest

- Runs automatically at **09:00 UTC** via the GitHub Actions workflow in `.github/workflows/daily_digest.yml`.
- Weekday rotation (UTC):
  - **Monday:** Summarize latest LLM retrieval methods.
  - **Tuesday:** Summarize latest multi-modal LLM research.
  - **Wednesday:** Summarize latest LLM fine-tuning and alignment papers.
  - **Thursday:** Summarize latest reinforcement learning or policy optimization in LLMs.
  - **Friday:** Summarize latest evaluation benchmarks for large language models.
  - **Saturday:** Summarize latest LLM efficiency and inference optimization research.
  - **Sunday:** Summarize latest applications of LLMs in reasoning and agents.
- Email delivery uses the following GitHub Secrets (all required unless noted):
  - `OPENAI_API_KEY`
  - `EMAIL_SENDER`
  - `EMAIL_PASSWORD`
  - `EMAIL_RECEIVER`
  - Optional overrides: `EMAIL_SMTP_HOST`, `EMAIL_SMTP_PORT`
- Each digest is sent as a multipart message (plain text + HTML) with the daily summary attached.

## ⚡ Automation

You can automate daily updates and summaries using:

GitHub Actions (recommended):
Run python src/scheduler_daily.py on a daily cron to:

- Fetch new papers

- Rebuild embeddings

- Generate a new LLM Research Digest in /reports/ (summaries generated with OpenAI gpt-4o-mini)


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
- [OpenAI API](https://platform.openai.com)
- [arXiv.org](https://arxiv.org)
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) — legacy summarizer used for Hugging Face regression tests.
- [Hugging Face Transformers](https://huggingface.co)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference) — used for model hosting and inference.
---

## 💬 About This Project

This project was developed as part of a **self-learning journey** to deepen my understanding of
**Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** systems.

With ChatGPT as a learning companion, I designed and implemented this project from scratch, covering data ingestion, vector search, model prompting, and automated summarization pipelines.
The goal is to bridge the gap between theory and real-world LLM deployment while building
a reusable framework for exploring GenAI research applications.

> 🧭 *This project is part of my ongoing exploration in AI/ML system design and Generative AI engineering.*

---
