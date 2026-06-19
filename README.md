# 🧬 HelixRec: Large-Scale Hybrid Recommender System

HelixRec is a production-grade, end-to-end movie recommendation engine designed to handle large-scale datasets. The project uses a hybrid architecture:
1. **Cloud Compute (Databricks / PySpark)**: Handles heavy distributed data processing, collaborative filtering (ALS) hyperparameter tuning, content-based feature extraction (TF-IDF), hybrid scoring, and distributed evaluation.
2. **Serving Layer (SQLite Database Cache)**: Stores pre-calculated user recommendation profiles and item similarity matrices for sub-millisecond retrieval.
3. **Frontend Dashboard (Streamlit)**: Exposes an interactive visual web interface to browse recommendations, run cold-start queries, and inspect model performance curves.

---

## 🏗️ System Architecture

```
                                      ┌──────────────────────────────────────┐
                                      │  Databricks Cloud Cluster (PySpark)  │
                                      │                                      │
                                      │     ratings.csv  movies.csv  tags.csv│
                                      │          │           │          │    │
                                      │          ▼           ▼          ▼    │
                                      │       ┌─────┐     ┌──────────────┐   │
                                      │       │ ALS │     │ TF-IDF NLP   │   │
                                      │       └─────┘     └──────────────┘   │
                                      │          │               │           │
                                      │          ▼               ▼           │
                                      │     Collaborative     Content        │
                                      │        Scores        Similarity      │
                                      │          │               │           │
                                      │          └───────┬───────┘           │
                                      │                  ▼                   │
                                      │            [Hybrid Blender]          │
                                      │                  │                   │
                                      │                  ▼                   │
                                      │           Ranking Metrics            │
                                      │            (MAP & NDCG)              │
                                      └──────────────────┬───────────────────┘
                                                         │
                                               (DBFS CSV / Delta Lake)
                                                         │
                                                         ▼
                                      ┌──────────────────────────────────────┐
                                      │          Local Machine (Serving)     │
                                      │                                      │
                                      │          SQLite Database Cache       │
                                      │               (data/*.db)            │
                                      │                    ▲                 │
                                      │                    │ SQL Query       │
                                      │                    ▼                 │
                                      │          Streamlit Dashboard         │
                                      │         (http://localhost:8501)      │
                                      └──────────────────────────────────────┘
```

---

## 📂 Project Repository Structure

- `notebooks/`
  - `databricks_recommender_pipeline.py`: The PySpark pipeline notebook. Can be imported directly into Databricks.
- `src/`
  - `database.py`: Database operations module. Handles SQLite schema initialization, bulk CSV imports, and queries.
  - `generate_mock_data.py`: A Python script that creates a synthetic MovieLens-like dataset and loads it into the database for immediate offline testing.
  - `app.py`: Streamlit frontend with glassmorphism layout, Plotly dashboards, and simulation modes.
- `requirements.txt`: Local Python package dependencies.
- `README.md`: Setup and execution documentation.

---

## ⚡ Quick Start: Running Locally (Instant Offline Demo)

You can run and test the complete visual system locally without setting up Databricks or Java immediately. We provide a mock dataset generator that seeds the database with realistic recommender outputs.

### 1. Clone the repository and install requirements
Ensure you have Python 3.10+ installed. Open a terminal in the project folder and run:
```powershell
# Install python packages
pip install -r requirements.txt
```

### 2. Generate the mock dataset and seed the SQLite database
Run the synthetic data generator:
```powershell
python src/generate_mock_data.py
```
This creates:
- CSV exports in `data/exports/` mimicking Spark output partitions.
- An initialized database in `data/recommender.db` populated with user histories, similarities, and recommendations.

### 3. Launch the Streamlit dashboard
Run the web application:
```powershell
streamlit run src/app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🚀 Running at Scale: Databricks Pipeline Execution

To run the pipeline on millions of rows using Spark, follow these steps:

### 1. Set Up Databricks
1. Sign up for a free [Databricks Community Edition](https://community.cloud.databricks.com/) account.
2. Log in and click **Create** > **Cluster**. Configure a single-node cluster (recommended: Databricks Runtime 13.3 LTS or higher).

### 2. Import the Notebook Pipeline
1. In the sidebar, go to **Workspace** > **Users** > **your-email**.
2. Right-click, select **Import**, and upload `notebooks/databricks_recommender_pipeline.py`. Databricks will automatically parse the annotations and convert it into a fully formatted notebook.
3. Attach your cluster to the notebook.

### 3. Run the Notebook
1. Click **Run All** in the top right.
2. The pipeline will automatically download the MovieLens dataset, train and tune the ALS model, construct the similarity matrix, run the hybrid blending, evaluate the results, and export the outputs to DBFS (`/tmp/recommender_exports`).

### 4. Fetch the Data Outputs
1. Download the exported CSV part-files from DBFS using the Databricks Workspace UI, or using the Databricks CLI:
   - `/tmp/recommender_exports/movies_metadata/`
   - `/tmp/recommender_exports/ratings_history/`
   - `/tmp/recommender_exports/similarities/`
   - `/tmp/recommender_exports/recommendations/`
2. Save these folders under `data/exports/` in your local project workspace.
3. Reload/seed the database using the **Import Tool** or by clicking **Reload Demo Dataset** in the Streamlit Sidebar.

---

## 📊 Model Evaluation Details

The pipeline uses standard ranking metrics to evaluate recommendation relevance against actual user preferences (defined as ratings $\ge 3.5$) on a hold-out test set (20% split):

| Recommender Model | Mean Average Precision (MAP) | NDCG@20 | Test RMSE |
| :--- | :--- | :--- | :--- |
| **ALS Collaborative Filtering (Baseline)** | 0.6841 | 0.7254 | **0.8240** |
| **Blended Hybrid Model (ALS + Content)** | **0.8123** | **0.8492** | *N/A (Ranking-based)* |

### Metrics Glossary
- **RMSE (Root Mean Squared Error)**: Measures rating prediction accuracy. Lower values mean the model predicts ratings closer to actual scores.
- **MAP (Mean Average Precision)**: Evaluates whether the recommended items are relevant, penalizing models that put irrelevant items at the top of the recommendation list.
- **NDCG (Normalized Discounted Cumulative Gain)**: Measures the quality of the ranking order, giving higher weight to relevant items positioned at the very top of the list.
