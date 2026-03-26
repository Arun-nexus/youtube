# 🎥 YouTube Comment Analyzer

An AI-powered system that extracts and analyzes YouTube comments to provide sentiment insights and trends directly within the browser.

---

## 🚀 Overview

This project enables users to analyze YouTube video comments in real-time using machine learning. It integrates a backend API, NLP models, and a browser-based interface to deliver insights like sentiment distribution and trends.

---

## ✨ Features

* 📥 Fetch comments from any YouTube video
* 🧠 Sentiment analysis (Positive / Neutral / Negative)
* 📊 Interactive dashboard (charts & trends)
* 🌐 Browser-based UI overlay
* ⚡ FastAPI backend for processing
* 🐳 Dockerized deployment

---

## 🖼️ Demo

* Real-time sentiment breakdown (Pie Chart)
* Monthly sentiment trend visualization
* Integrated UI panel inside browser

---

## 🏗️ Architecture

```text
YouTube Video Page
        ↓
Frontend / Extension UI
        ↓
FastAPI Backend
        ↓
YouTube API (comments)
        ↓
NLP Model (Sentiment Analysis)
        ↓
Processed Insights (Charts)
```

---

## ⚙️ Tech Stack

* **Frontend/UI**: JavaScript (Browser overlay / extension)
* **Backend**: FastAPI (Python)
* **ML/NLP**: NLTK / Scikit-learn / Custom model
* **Visualization**: Chart.js / custom UI
* **API**: YouTube Data API
* **Deployment**: Docker, AWS (optional)

---

## 📦 Project Structure

```bash
.
├── app.py
├── requirement.txt
├── src/
│   ├── fetch_comments.py
│   ├── preprocess.py
│   ├── sentiment_model.py
│   └── utils.py
├── extension/
├── dockerfile
├── logs/
└── notebooks/
```

---

## 🧪 How It Works

1. User opens a YouTube video
2. Extension/UI captures video context
3. Backend fetches comments using YouTube API
4. Comments are cleaned and processed
5. Sentiment model classifies each comment
6. Results are displayed as charts

---

## 🔧 Setup & Installation

### 1. Clone repository

```bash
git clone https://github.com/Arun-nexus/youtube.git
cd youtube
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Run backend

```bash
uvicorn app:app --reload
```

---

## 🐳 Docker Setup

```bash
docker build -t youtube-analyzer .
docker run -p 8000:8000 youtube-analyzer
```

---

## 🔑 Environment Variables

```env
YOUTUBE_API_KEY=your_api_key
```

---

## ⚠️ Known Issues & Limitations

* YouTube API quota limits may restrict large-scale analysis
* Real-time analysis depends on API response speed
* Sentiment accuracy depends on dataset quality
* UI overlay may vary across different YouTube layouts

---

## 🔐 Security Considerations

* Never expose API keys in public repositories
* Use `.env` files for secrets
* Rotate keys if compromised

---

## 📈 Future Improvements

* Real-time streaming of comments
* Advanced NLP (transformer-based models)
* User dashboard with history
* Deployment with autoscaling
* Better UI/UX and animations

---

## 🎯 Learning Outcomes

* Built an end-to-end NLP pipeline
* Integrated browser UI with backend APIs
* Worked with real-world APIs (YouTube Data API)
* Implemented data visualization for insights
* Deployed containerized applications

---

## 🧑‍💻 Author

Arun
Aspiring AI/ML Engineer

---
