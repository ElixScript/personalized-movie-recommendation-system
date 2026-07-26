# 🎬 Movie Recommendation System

> Boosting engagement on a streaming platform with personalized recommendations — an end-to-end data-science case study, from business framing through EDA, five modeling families, task-appropriate evaluation, and business translation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/data-MovieLens-informational)](https://grouplens.org/datasets/movielens/)

---

## 📌 The Problem

A subscription streaming service lives and dies by **engagement**. When users can't quickly find something worth watching, session length drops and churn rises. The current experience shows *everyone the same "Popular Now" row* — no personalization, and thousands of relevant niche titles stay invisible.

**Goal:** build and rigorously evaluate a recommender that surfaces the *right* titles per user, to lift click-/watch-through on the recommendation shelf and, downstream, retention.

## 📚 Dataset

[**MovieLens `ml-latest-small`**](https://grouplens.org/datasets/movielens/) (GroupLens Research) — ~100,000 ratings from ~610 users on ~9,700 movies, plus genres and user tags. The notebook **downloads it automatically** on first run; no manual setup required.

## 🧠 Approach

Five recommender families of increasing sophistication, each targeting a weakness of the last:

| # | Model | Idea |
|---|-------|------|
| 1 | **Popularity** (baseline) | IMDb-style Bayesian weighted rating — a strong, non-personalized benchmark |
| 2 | **Content-based** | TF-IDF over genres + tags → user taste profiles; handles cold-start |
| 3 | **Collaborative Filtering** | Item-based & user-based neighborhood models (cosine, top-K) |
| 4 | **Matrix Factorization (SVD)** | Latent factors on **bias-adjusted residuals** (Koren regularization) |
| 5 | **Hybrid** | Rank-blend of the complementary strengths |

Crucially, models are judged on **two distinct tasks** with the metrics that fit each:
- **Rating prediction** → RMSE / MAE
- **Top-*N* ranking** (what the user actually sees) → Precision@K, Recall@K, NDCG@K, and catalog coverage

## 📊 Key Results

**Rating prediction** — SVD wins:

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| **SVD (MF)** | **0.831** | **0.642** |
| Baseline (bias) | 0.840 | 0.650 |
| User-based CF | 0.869 | 0.661 |
| Item-based CF | 0.885 | 0.661 |

**Top-10 ranking** — neighborhood CF & the Hybrid win:

| Model | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
|---|---|---|---|---|
| **User-based CF** | **0.153** | **0.142** | **0.208** | 0.130 |
| Hybrid | 0.142 | 0.126 | 0.192 | 0.133 |
| Item-based CF | 0.134 | 0.124 | 0.183 | **0.529** |
| Popularity | 0.106 | 0.088 | 0.133 | 0.019 |
| SVD (MF) | 0.097 | 0.080 | 0.127 | 0.078 |
| Content-based | 0.014 | 0.011 | 0.017 | 0.491 |

### 💡 Headline insights

- **The best rating-predictor is *not* the best ranker.** SVD minimizes RMSE, but User-based CF produces the best-ranked lists. *Select the model on the metric that matches the product surface* — here, ranking.
- **Personalization beats the status quo.** The best personalized rankers outrank the Popularity baseline on NDCG@10 — the business case for building this at all.
- **Coverage is a business lever.** Popularity shows the same ~2% of the catalog to everyone; Item-based CF and Content-based reach ~50%+, exposing the profitable long tail. Content-based additionally handles **cold-start** items.
- **Ship a Hybrid.** It lands near the top on ranking and its blend weights directly trade ranking for reach — the most tunable, production-friendly profile.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the notebook (data downloads automatically on first run)
jupyter lab movie_recommendation_system.ipynb
```

Then **Run All** — the notebook executes top-to-bottom with no manual steps.

Get recommendations for any user directly:

```python
recommend(user_id=1, n=10, model="hybrid")
# → ranked DataFrame of titles, genres, and relevance scores
```

## 🗂️ Repository Structure

```
.
├── movie_recommendation_system.ipynb   # main notebook (executed, with outputs)
├── requirements.txt                    # pinned dependencies
├── README.md
├── .gitignore
└── data/                               # MovieLens files (auto-downloaded, gitignored)
```

## 🔭 Limitations & Future Work

Small offline dataset (~610 users); offline metrics are a proxy for real engagement, so the natural next step is an **online A/B test**. Roadmap: learning-to-rank (BPR/WARP), regularized MF via SGD/ALS, implicit-feedback modeling, neural/sequence models (two-tower, GRU4Rec, SASRec), and productionization with ANN retrieval (FAISS). See the notebook's final section for details.

## 📄 Data Attribution

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS). [DOI:10.1145/2827872](https://doi.org/10.1145/2827872)

---

*Built as a data-science portfolio project. The pipeline is catalog-agnostic — the same techniques power **product**, music, and news recommendations.*
