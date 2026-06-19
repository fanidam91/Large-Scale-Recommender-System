# Databricks notebook source
# MAGIC %md
# MAGIC # Large-Scale Recommender System Pipeline
# MAGIC This notebook runs on Databricks (Spark) and builds a hybrid recommender system combining:
# MAGIC 1. **Collaborative Filtering** using PySpark MLlib's Alternating Least Squares (ALS).
# MAGIC 2. **Content-Based Filtering** using TF-IDF on movie genres and user-generated tags.
# MAGIC 3. **Hybrid Recommendation Engine** which blends collaborative scores with item similarity profiles.
# MAGIC 
# MAGIC It evaluates models using distributed ranking metrics (NDCG, MAP) and saves the results to DBFS/Delta Lake for serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Import Libraries and Initialize Session
# MAGIC Import standard PySpark MLlib and SQL utilities.

# COMMAND ----------

import os
import urllib.request
import zipfile
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType

# Spark ML modules
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics

# Initialize Spark Session (automatically managed in Databricks, but set up for compatibility)
spark = SparkSession.builder \
    .appName("LargeScaleRecommenderSystem") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Download and Ingest MovieLens Dataset
# MAGIC We download the MovieLens `ml-latest-small` dataset which contains ratings and user-applied movie tags, making it perfect for hybrid recommenders.

# COMMAND ----------

# Setup directories in DBFS
import urllib.request
import zipfile
import os

dbfs_dir = "/tmp/recommender_data"
dbutils.fs.mkdirs(dbfs_dir)

import uuid
unique_id = str(uuid.uuid4())[:8]
local_dir = f"/tmp/helixrec_temp_{unique_id}"
os.makedirs(local_dir, exist_ok=True)

local_zip = os.path.join(local_dir, "ml-latest-small.zip")
extract_path = os.path.join(local_dir, "ml-extract")

# Download the dataset
dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
print(f"Downloading dataset from {dataset_url}...")
urllib.request.urlretrieve(dataset_url, local_zip)

# Extract file locally
print("Extracting dataset...")
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Move extracted files to DBFS
print("Moving files to DBFS...")
extracted_folder_name = os.listdir(extract_path)[0]  # ml-latest-small
src_folder = os.path.join(extract_path, extracted_folder_name)

for file_name in os.listdir(src_folder):
    local_file_path = f"file:{os.path.join(src_folder, file_name)}"
    dbfs_file_path = f"dbfs:{dbfs_dir}/{file_name}"
    dbutils.fs.mv(local_file_path, dbfs_file_path)
    print(f"Moved {file_name} to DBFS.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Load DataFrames and Inspect Schema
# MAGIC We load the CSV files from DBFS and parse them into structured Spark DataFrames.

# COMMAND ----------

# Load DataFrames
ratings_df = spark.read.csv(f"{dbfs_dir}/ratings.csv", header=True, inferSchema=True)
movies_df = spark.read.csv(f"{dbfs_dir}/movies.csv", header=True, inferSchema=True)
tags_df = spark.read.csv(f"{dbfs_dir}/tags.csv", header=True, inferSchema=True)

print("Ratings Data:")
ratings_df.show(5)
print("Movies Data:")
movies_df.show(5)
print("Tags Data:")
tags_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Collaborative Filtering - ALS Model
# MAGIC We split ratings into training and testing sets, then build an ALS model. We use grid-search cross-validation to select parameters.

# COMMAND ----------

# Train/Test Split (80/20)
(training_df, test_df) = ratings_df.randomSplit([0.8, 0.2], seed=42)
training_df.cache()
test_df.cache()

# Define ALS Model
# Note: coldStartStrategy="drop" drops users/items in the test set not seen in training during evaluation
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    coldStartStrategy="drop"
)

# Hyperparameter Grid
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [8, 12, 16]) \
    .addGrid(als.regParam, [0.05, 0.1, 0.15]) \
    .build()

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# Cross Validation
cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    seed=42
)

print("Training ALS model with 3-fold cross validation...")
cv_model = cv.fit(training_df)
best_model = cv_model.bestModel

print(f"Best Model Rank: {best_model.rank}")
print(f"Best Model regParam: {best_model._java_obj.parent().getRegParam()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Evaluate ALS Recommendations
# MAGIC Compute RMSE and test prediction quality.

# COMMAND ----------

predictions = best_model.transform(test_df)
rmse = evaluator.evaluate(predictions)
print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Content-Based Filtering - TF-IDF
# MAGIC We extract rich movie representations by combining movie genres and user tags. We then apply Spark's NLP TF-IDF features to represent each movie as a term-frequency vector.

# COMMAND ----------

# Group tags by movieId and merge into a space-separated string
tags_grouped = tags_df.groupBy("movieId") \
    .agg(F.concat_ws(" ", F.collect_list("tag")).alias("tags"))

# Join movies with tags and replace pipe symbol in genres with space
movies_enriched = movies_df.join(tags_grouped, on="movieId", how="left") \
    .withColumn("genres_clean", F.regexp_replace(F.col("genres"), r"\|", " ")) \
    .withColumn("text_features", F.concat_ws(" ", F.col("genres_clean"), F.coalesce(F.col("tags"), F.lit("")))) \
    .select("movieId", "title", "text_features")

movies_enriched.cache()
print("Enriched Movie Documents:")
movies_enriched.show(5, truncate=False)

# Spark NLP Pipeline: Tokenizer -> HashingTF -> IDF
tokenizer = Tokenizer(inputCol="text_features", outputCol="words")
words_df = tokenizer.transform(movies_enriched)

hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2000)
featurized_df = hashing_tf.transform(words_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df).select("movieId", "title", "features")

tfidf_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Movie-to-Movie Cosine Similarities
# MAGIC We normalize the TF-IDF feature vectors and compute cosine similarities using a self-join dot product. To keep storage small, we save only the top 20 most similar movies for each movie.

# COMMAND ----------

# Normalize vectors to unit L2 norm
normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
norm_tfidf_df = normalizer.transform(tfidf_df).select("movieId", "norm_features")

# Convert to vector dot product computation
# To scale, we perform a join and calculate dot product via a Python UDF.
# Since L2 norm = 1, dot product equals cosine similarity.
@F.udf(returnType=DoubleType())
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return float(v1.dot(v2))

# Rename columns to perform self-join
df_left = norm_tfidf_df.select(F.col("movieId").alias("movie_a"), F.col("norm_features").alias("vec_a"))
df_right = norm_tfidf_df.select(F.col("movieId").alias("movie_b"), F.col("norm_features").alias("vec_b"))

# Filter to avoid redundancy and compute similarity
similarities_all = df_left.join(df_right, df_left.movie_a < df_right.movie_b) \
    .withColumn("similarity", cosine_similarity("vec_a", "vec_b")) \
    .filter("similarity > 0.1") \
    .select("movie_a", "movie_b", "similarity")

# Keep only the top 20 similar movies per movie
window_spec = Window.partitionBy("movie_a").orderBy(F.desc("similarity"))
top_similarities = similarities_all \
    .withColumn("rank", F.row_number().over(window_spec)) \
    .filter("rank <= 20") \
    .select(
        F.col("movie_a").alias("movieId"),
        F.col("movie_b").alias("similarMovieId"),
        "similarity"
    )

# Union the symmetric relation (if A is similar to B, then B is similar to A)
symmetric_similarities = similarities_all \
    .select(F.col("movie_b").alias("movieId"), F.col("movie_a").alias("similarMovieId"), "similarity") \
    .withColumn("rank", F.row_number().over(Window.partitionBy("movieId").orderBy(F.desc("similarity")))) \
    .filter("rank <= 20")

# Final combined similarity matrix
movie_similarity_matrix = top_similarities.union(symmetric_similarities).distinct()
movie_similarity_matrix.cache()

print("Movie-to-Movie Cosine Similarities:")
movie_similarity_matrix.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Hybrid Recommendation Logic
# MAGIC The hybrid recommender calculates user recommendations by blending ALS collaborative ratings with content-based movie similarities.
# MAGIC For user $u$ and item $i$, we define:
# MAGIC 
# MAGIC $$\text{Score}_{\text{Hybrid}}(u, i) = w \cdot \text{Score}_{\text{ALS}}(u, i) + (1-w) \cdot \text{Score}_{\text{Content}}(u, i)$$
# MAGIC 
# MAGIC Where $\text{Score}_{\text{Content}}(u, i)$ represents the weighted average similarity of movie $i$ to other movies that user $u$ has rated highly ($> 3.5$).

# COMMAND ----------

# 1. Generate top 50 ALS recommendations for all users
als_recs = best_model.recommendForAllUsers(50) \
    .withColumn("rec", F.explode("recommendations")) \
    .select("userId", F.col("rec.movieId").alias("movieId"), F.col("rec.rating").alias("als_score"))

# 2. Extract highly rated movies for each user from training set
user_likes = training_df.filter("rating >= 3.5") \
    .select("userId", F.col("movieId").alias("liked_movieId"), "rating")

# 3. Find candidate movies similar to the user's liked movies
content_candidates = user_likes.join(movie_similarity_matrix, user_likes.liked_movieId == movie_similarity_matrix.movieId) \
    .groupBy("userId", "similarMovieId") \
    .agg(F.max("similarity").alias("content_score")) \
    .select("userId", F.col("similarMovieId").alias("movieId"), "content_score")

# 4. Join collaborative and content candidates
hybrid_recs_all = als_recs.join(content_candidates, on=["userId", "movieId"], how="outer") \
    .na.fill(0.0)

# 5. Compute Hybrid Score (weighting ALS 0.6 and Content-based 0.4)
w_als = 0.6
w_content = 0.4
hybrid_recs = hybrid_recs_all \
    .withColumn("hybrid_score", F.lit(w_als) * F.col("als_score") + F.lit(w_content) * F.col("content_score"))

# Get top 20 recommendations per user
user_window = Window.partitionBy("userId").orderBy(F.desc("hybrid_score"))
final_recommendations = hybrid_recs \
    .withColumn("rank", F.row_number().over(user_window)) \
    .filter("rank <= 20") \
    .select("userId", "movieId", "als_score", "content_score", "hybrid_score")

final_recommendations.cache()
print("Final Hybrid Recommendations for Users:")
final_recommendations.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Distributed Evaluation (NDCG, MAP)
# MAGIC We evaluate the models on the hold-out test set. We construct lists of actual highly rated items (rating >= 3.5) per user and compare them to the top-20 generated recommendations from ALS and our Hybrid model.

# COMMAND ----------

# 1. actual highly-rated movies per user in the test set
test_actual = test_df.filter("rating >= 3.5") \
    .groupBy("userId") \
    .agg(F.collect_list("movieId").alias("actual_movies"))

# 2. predicted movies for ALS
als_recs_list = als_recs \
    .withColumn("rank", F.row_number().over(Window.partitionBy("userId").orderBy(F.desc("als_score")))) \
    .filter("rank <= 20") \
    .groupBy("userId") \
    .agg(F.collect_list("movieId").alias("predicted_als"))

# 3. predicted movies for Hybrid
hybrid_recs_list = final_recommendations \
    .groupBy("userId") \
    .agg(F.collect_list("movieId").alias("predicted_hybrid"))

# Join and prepare RDD for RankingMetrics
eval_rdd_als = als_recs_list.join(test_actual, "userId") \
    .select("predicted_als", "actual_movies") \
    .rdd.map(lambda row: (row[0], row[1]))

eval_rdd_hybrid = hybrid_recs_list.join(test_actual, "userId") \
    .select("predicted_hybrid", "actual_movies") \
    .rdd.map(lambda row: (row[0], row[1]))

# Compute Spark RankingMetrics
metrics_als = RankingMetrics(eval_rdd_als)
metrics_hybrid = RankingMetrics(eval_rdd_hybrid)

print(f"Collaborative Filtering (ALS) MAP: {metrics_als.meanAveragePrecision:.4f}")
print(f"Collaborative Filtering (ALS) NDCG@20: {metrics_als.ndcgAt(20):.4f}")
print("---")
print(f"Hybrid Recommender MAP: {metrics_hybrid.meanAveragePrecision:.4f}")
print(f"Hybrid Recommender NDCG@20: {metrics_hybrid.ndcgAt(20):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Export Datasets to DBFS / Local
# MAGIC We write out the outputs as single CSV partitions so that we can easily download them and load them into our local SQLite/PostgreSQL caching database.

# COMMAND ----------

# Coalesce to 1 partition for easy single-file download
output_dir = "/tmp/recommender_exports"
dbutils.fs.mkdirs(output_dir)

# Save recommendations
final_recommendations.coalesce(1).write.csv(f"{output_dir}/recommendations", header=True, mode="overwrite")
# Save movie similarities
movie_similarity_matrix.coalesce(1).write.csv(f"{output_dir}/similarities", header=True, mode="overwrite")
# Save movie metadata
movies_enriched.coalesce(1).write.csv(f"{output_dir}/movies_metadata", header=True, mode="overwrite")
# Save raw user ratings
ratings_df.coalesce(1).write.csv(f"{output_dir}/ratings_history", header=True, mode="overwrite")

print(f"Export completed. Files stored under {output_dir} in DBFS.")
