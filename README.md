# 📚 RAG Assistant — Automated Research Summarizer for LLM Papers
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![arXiv Powered](https://img.shields.io/badge/Data-arXiv-orange)](https://arxiv.org)


RAG Assistant is an end-to-end **automated Retrieval-Augmented Generation (RAG)** system that continuously fetches the newest **LLM-related research papers** from **arXiv**, processes them on the fly, and enables **interactive question-answering and weekly summarization** of emerging AI research trends (with an optional high-frequency daily mode).  

> 📉 **Why weekly by default?** We observed that the general LLM query and topic-specific prompts rarely surface enough fresh papers every single day to produce meaningful digests. Switching the automation to a weekly cadence yields richer, less repetitive summaries while still allowing on-demand daily runs when research volume spikes.

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
- **Automated Research Digests** – Generates concise “LLM Research Digest” summaries on a weekly cadence (with an optional daily fast mode for high-volume periods).
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
| Automation | `cron` / GitHub Actions (weekly refresh, daily optional) |

---

## 🧩 Project Structure

```
rag-assistant/
├── src/
│ ├── fetch_arxiv.py # Search & fetch latest LLM papers from arXiv
│ ├── ingest_stream.py # Stream, parse, and chunk PDF text
│ ├── embed_index.py # Build FAISS index & store metadata
│ ├── rag_answer.py # Query, retrieve, and generate LLM-based answers
│ ├── scheduler_daily.py # Optional daily index refresh + digest summary (shared pipeline)
│ ├── scheduler_weekly.py # Weekly wrapper (default cadence) that reuses the shared scheduler logic
│ └── utils/clean.py # Optional text cleanup helpers
├── indexes/ # FAISS index + metadata JSON
├── reports/ # LLM Digest markdown files
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
python src/rag_answer.py --provider openai --model gpt-4o-mini --query "Summarize the latest research on large language models (methods, training, alignment/safety, evaluation, efficiency, multimodal, and applications). Provide concise, linked bullets."
```

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
OPENAI_API_KEY=sk-xxxx python src/rag_answer.py --provider openai --model gpt-4o-mini --query "Summarize the latest research on large language models (methods, training, alignment/safety, evaluation, efficiency, multimodal, and applications). Provide concise, linked bullets."

# Legacy Hugging Face testing only (facebook/bart-large-cnn)
HF_TOKEN=hf_xxx python src/rag_answer.py --provider hf --query "Summarize latest LLM retrieval methods."

> Note: `--provider hf` is kept only for regression testing with `facebook/bart-large-cnn`. All production runs should use `--provider openai`.
```

### 5. Run Weekly Digest (default)
```bash
python src/scheduler_weekly.py --days-back 7 --dry-run
```

Full scheduler run (writes reports/YYYY-MM-DD-llm-digest.md): 

```bash
OPENAI_API_KEY=sk-xxxx python src/scheduler_weekly.py --days-back 7
```

Need a higher-frequency readout?
```bash
python src/scheduler_daily.py --days-back 1 --dry-run
```


</details>

## 📬 Automated Research Digest

- Runs automatically at **16:00 UTC each Monday** via the GitHub Actions workflow in `.github/workflows/daily_digest.yml` (switch the command to `scheduler_daily.py` if you want the higher-frequency mode).
- Uses a single **generic LLM mega-query** (`GENERIC_LLM_QUERY`) that covers methods, training, alignment/safety, evaluation, efficiency, multimodal work, and applications in one pass.
- Applies strict deduplication (arXiv id root / DOI / near-identical titles) before emitting HTML bullets so each paper appears once with its newest canonical link.
- Weekly emails and HTML reports label the reporting window as `MMM DD, YYYY - MMM DD, YYYY (XTH WEEK)` so you can see coverage at a glance.
- Email delivery uses the following GitHub Secrets (all required unless noted):
  - `OPENAI_API_KEY`
  - `EMAIL_SENDER`
  - `EMAIL_PASSWORD`
  - `EMAIL_RECEIVER`
  - Optional overrides: `EMAIL_SMTP_HOST`, `EMAIL_SMTP_PORT`
- Each digest is sent as a multipart message (plain text + HTML) with the weekly summary attached (the daily variant shares the same format when invoked manually).

## ⚡ Automation

You can automate weekly updates and summaries using:

GitHub Actions (recommended):
Run `python src/scheduler_weekly.py` on a weekly cron to:

- Fetch new papers
- Rebuild embeddings
- Generate a new LLM Research Digest in /reports/ (summaries generated with OpenAI gpt-4o-mini)


### Example GitHub Action (cron.yaml)
```yaml
name: Weekly Digest
on:
  schedule:
    - cron: "0 14 * * MON"  # every Monday at 14:00 UTC
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
      - run: python src/scheduler_weekly.py
```

Need faster iterations? Replace the final step with `python src/scheduler_daily.py` and adjust the cron to `0 14 * * *`.

## Modes & Branches

- `main` (stable): runs **Auto Generic** weekly — fetches recent LLM papers; if a topic is requested and relevant sources ≥ N, it applies a topic lens, otherwise falls back to generic safely.
- `feature/topic-digests`: current **multi-topic** prototype (may drift off-topic).
- `feature/auto-generic`: in-progress refactor for **Auto mode** with hybrid retrieval, reranker, and guardrails.

## 📈 Roadmap

- Add reranking model for improved retrieval precision

- Implement hybrid (keyword + vector) retrieval

- Support multi-source ingestion (CORE, Hugging Face datasets)

- Enhance summarization metrics (faithfulness, recall)

- Deploy Weekly Digest to Slack / Discord

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
