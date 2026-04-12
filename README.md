# 🎯 Kaggle Job Recommendation System

A production-grade **AI-enhanced ML pipeline** for job recommendations using:
- **TF-IDF vectorization** with bigram support for text feature extraction
- **Cosine similarity** for job-skill matching
- **Gemma 3 AI** (via Docker/Ollama) for intelligent explanations
- **Real Kaggle dataset** (50,000 actual job postings)
- **Supervised & Unsupervised ML** with clustering visualizations
- **Automatic evaluation** with quality metrics

Everything runs end-to-end with a single command or through the **Jupyter Notebook**.

---

## 🚀 Quick Start

### Option A: Jupyter Notebook (Recommended)
```bash
pip install -r requirements.txt
jupyter notebook Job_Recommendation_System.ipynb
```
Run all cells top-to-bottom. The notebook is fully self-contained.

### Option B: Python Script
```bash
pip install -r requirements.txt
python example_kaggle.py
```

### Option C: With Gemma 3 AI (Docker)
```bash
# Start Ollama with Gemma 3
docker compose up -d
docker exec ollama-gemma3 ollama pull gemma3

# Run the pipeline
python example_kaggle.py
```

---

## 📊 8-Phase Execution Flow

| Phase | Description | Output |
|-------|-------------|--------|
| **1. Load Dataset** | Loads 50,000 real job postings from Kaggle | Dataset shape, columns |
| **2. Preprocessing** | Handle missing values, clean text | Rows removed count |
| **3. Feature Engineering** | TF-IDF vectorization with unigrams + bigrams | Feature dimensions |
| **4. User Input** | User enters comma-separated skills | Skill list |
| **5. Recommendations** | Cosine similarity → Top 5 jobs | Ranked job list |
| **6. Evaluation** | Quality metrics (avg score, percentile) | Quality grade |
| **7. Visualization** | Generate publication-quality graphs | 3+ PNG files |
| **8. Analysis** | Detailed insights and AI explanations | Career insights |

---

## 📈 Example Output

```
🏆 Top Recommendation:
  • Job: Senior Data Scientist
  • Company: Tech Corp
  • Match Score: 0.793 (79.3%)
  • Skill Overlap: 75%
  • Matched Skills: python, machine learning, sql

💡 Key Insights:
  • Average match quality: 75.1%
  • System inference time: 3.27ms
  • Total jobs analyzed: 50,000
```

---

## 🤖 Gemma 3 AI Integration

The system optionally uses **Google's Gemma 3** (4B parameter model, ~3.1GB) via Docker for:
- Natural language explanations for each recommendation
- Skill gap analysis
- Career direction advice

**Requirements:**
- Docker Desktop installed and running
- ~4GB free disk space for the model

**Setup:**
```bash
docker compose up -d                              # Start Ollama container
docker exec ollama-gemma3 ollama pull gemma3       # Download Gemma 3 (~3.1GB)
```

> The system gracefully degrades if Docker/Gemma is not running — all TF-IDF features work without it.

---

## 📊 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Vectorization** | TF-IDF with bigrams (scikit-learn) |
| **Similarity** | Cosine Similarity |
| **AI Model** | Gemma 3 4B via Ollama/Docker |
| **Dataset** | Kaggle (50,000 jobs) |
| **ML Analysis** | K-Means, PCA, t-SNE, Classification |
| **Evaluation** | Precision@K, Recall@K, MRR@K |
| **Visualization** | Matplotlib + Seaborn |
| **Data Processing** | Pandas + NumPy |
| **Notebook** | Jupyter (Python 3 kernel) |

---

## 📁 Project Structure

```
AIDS-ML-Project/
├── README.md                              # Project overview (this file)
├── USER.md                                # User guide & documentation
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Gemma 3 AI container
├── docker-compose.yml                     # Docker orchestration
├── .gitignore                             # Git exclusions
│
├── Job_Recommendation_System.ipynb        # ⭐ Main Jupyter Notebook
├── example_kaggle.py                      # Python script alternative
│
├── data/
│   └── job_recommendation_dataset.csv     # 50,000 jobs (Kaggle)
│
├── models/                                # Trained model artifacts
│   ├── job_recommender_kaggle_vectorizer.pkl
│   ├── job_recommender_kaggle_data.pkl
│   └── job_recommendation_model.pkl
│
├── graphs/                                # Generated visualizations
│   ├── 01_top_5_recommendations.png
│   ├── 02_similarity_distribution.png
│   └── 03_top_skills.png
│
└── ml_module/                             # Core ML modules
    ├── __init__.py                        # Package init
    ├── job_recommendation_kaggle.py       # TF-IDF recommendation engine
    ├── kaggle_adapter.py                  # Dataset loading & adaptation
    ├── evaluation_kaggle.py               # Evaluation metrics
    ├── visualization_kaggle.py            # Graph generation
    └── gemma_recommender.py               # Gemma 3 AI integration
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip
- Docker Desktop (optional, for Gemma 3 AI)

### Steps
```bash
# Clone the repository
git clone https://github.com/aryangaikwad-966/AIDS--ML--Project.git
cd AIDS--ML--Project

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook Job_Recommendation_System.ipynb

# (Optional) Start Gemma 3 AI
docker compose up -d
docker exec ollama-gemma3 ollama pull gemma3
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **TF-IDF Dimensions** | ~2000+ (with bigrams) |
| **Average Top-5 Score** | 0.65–0.79 |
| **Best Match Percentile** | 100% |
| **Inference Time** | ~3–5ms |
| **Dataset Size** | 50,000 jobs |
| **Gemma 3 Enhancement** | +15s per recommendation |

---

## 🔍 How It Works

```
User Input: "Python, Machine Learning, SQL"
    ↓
TF-IDF Vectorization (unigrams + bigrams)
    ↓
Cosine Similarity with 50,000 job vectors
    ↓
Top 5 ranked by similarity score
    ↓
Skill overlap analysis (comma-aware matching)
    ↓
(Optional) Gemma 3 AI explanations
    ↓
Results + Graphs + Career Insights
```

---

## 🎓 Academic Quality

- ✅ Complete end-to-end ML pipeline
- ✅ Real dataset (50K jobs from Kaggle)
- ✅ Proper data preprocessing
- ✅ Feature engineering (TF-IDF with bigrams)
- ✅ Supervised & unsupervised learning
- ✅ AI model integration (Gemma 3)
- ✅ Docker containerization
- ✅ Professional visualizations
- ✅ Jupyter Notebook format
- ✅ Detailed analysis & insights

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: data/...csv` | Make sure you're in the project root directory |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Graphs not displaying | Check `graphs/` folder for saved PNGs |
| Docker not starting | Ensure Docker Desktop is running |
| Gemma 3 not found | Run `docker exec ollama-gemma3 ollama pull gemma3` |

---

## 👥 Team

**AIDS ML Project** — Artificial Intelligence & Data Science

---

**Status:** ✅ Production Ready  
**Version:** 3.0.0  
**Last Updated:** April 13, 2026
