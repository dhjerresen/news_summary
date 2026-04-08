# 📰 News Summary MLOps Pipeline

This project implements an end-to-end **MLOps pipeline** for automated news summarization using LLMs.
The system ingests real-world news data, processes and clusters articles, generates summaries, enriches them with external knowledge, and exposes results through a lightweight frontend.

The focus of the project is **operationalization, reproducibility, and deployment**, rather than model performance.

---

## 🚀 Project Overview

The pipeline performs the following steps:

1. **Data Ingestion**

   * Fetches latest news articles from an external API

2. **Preprocessing & Clustering**

   * Cleans and groups related articles into clusters

3. **LLM-based Summarization**

   * Generates summaries and extracts topics using an LLM

4. **Enrichment**

   * Adds contextual information via Wikipedia API

5. **Artifact Storage**

   * Stores all intermediate and final outputs per run

6. **Frontend Serving**

   * Displays latest results via GitHub Pages

---

## 🏗️ Pipeline Architecture

```
News API → Raw Data → Preprocessing → Clustering
         → LLM Summarization → Wikipedia Enrichment
         → Artifacts (JSON + logs)
         → GitHub Pages Frontend
```

---

## 📂 Repository Structure

```
app/
  ├── pipeline.py          # Main pipeline orchestration
  ├── world_news_api.py    # Data ingestion
  ├── preprocessing.py     # Cleaning & clustering
  ├── llm.py               # LLM summarization logic
  ├── wiki_api.py          # Wikipedia enrichment
  ├── utils.py             # Artifact handling & utilities

data/
  └── artifacts/           # Stored pipeline runs

docs/
  └── latest.json          # Latest run metadata (used by frontend)

frontend/
  └── (GitHub Pages site)

Dockerfile
docker-compose.yml
main.py                    # Entry point
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/dhjerresen/news_summary.git
cd news_summary
```

---

### 2. Create environment variables

Create a `.env` file:

```env
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_key
```

---

### 3. Install dependencies (local run)

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Run locally

```bash
python main.py
```

This will:

* Fetch latest news
* Process and cluster articles
* Generate summaries
* Save artifacts to `/data/artifacts/`
* Update `/docs/latest.json`

---

## 🐳 Running with Docker

### Build and run

Make sure Docker Desktop is running, then execute:

```bash
docker-compose up --build
```

This ensures a **reproducible and containerized execution environment**.

---

## 📦 Artifacts & Reproducibility

Each pipeline run generates a unique `run_id` and stores:

* `raw_news.json` → raw input data
* `processed_clusters.json` → clustered data
* `summaries.json` → generated summaries
* `failed_clusters.json` → failed processing cases
* `metadata.json` → run metadata
* `logs.txt` → execution logs

The latest run is tracked in:

```
docs/latest.json
```

This enables **full reproducibility and traceability** of results.

---

## 📊 Monitoring & Versioning

The system logs the following per run:

* Number of articles and clusters
* Successful vs failed summaries
* Wikipedia enrichment success rate
* Runtime and timestamps
* Model name and prompt version

This supports:

* Basic **pipeline monitoring**
* **Model and prompt version tracking**
* Debugging and evaluation

---

## 🌐 Frontend

A lightweight frontend is deployed via GitHub Pages:

👉 https://dhjerresen.github.io/news_summary/

The frontend displays:

* Latest pipeline run
* Key metrics
* Top summaries

---

## 🔄 Deployment Strategy

The system is designed to be:

* **Triggered manually** (via `main.py`)
* **Containerized** (Docker)
* Extendable to:

  * Scheduled runs (cron / GitHub Actions)
  * API-based triggering (e.g., FastAPI)

---

## 🧠 MLOps Considerations

This project demonstrates:

* End-to-end pipeline orchestration
* Artifact tracking and storage
* Reproducible execution (Docker)
* Monitoring via metadata logging
* Separation of pipeline components
* Deployment-ready structure

---

## ⚠️ Limitations

* Focus is on pipeline design, not model optimization
* LLM outputs may vary between runs
* No automated evaluation metrics (future work)

---

## 🔮 Future Improvements

* Add automated scheduling (GitHub Actions)
* Implement API endpoint for triggering runs
* Improve frontend visualization
* Add evaluation metrics for summary quality

---

## 📜 License

This project is for educational purposes as part of an MLOps exam assignment.

---