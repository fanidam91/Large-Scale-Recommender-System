import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

DB_PATH = os.path.join("data", "recommender.db")

def get_engine():
    """Returns a SQLAlchemy engine for the SQLite database."""
    os.makedirs("data", exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")

def init_db():
    """Initializes the database schema if tables do not exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create movies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY,
            title TEXT,
            text_features TEXT
        )
    """)

    # Create ratings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            userId INTEGER,
            movieId INTEGER,
            rating REAL,
            timestamp INTEGER,
            PRIMARY KEY (userId, movieId)
        )
    """)

    # Create similarities table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS similarities (
            movieId INTEGER,
            similarMovieId INTEGER,
            similarity REAL,
            PRIMARY KEY (movieId, similarMovieId)
        )
    """)

    # Create recommendations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            userId INTEGER,
            movieId INTEGER,
            als_score REAL,
            content_score REAL,
            hybrid_score REAL,
            PRIMARY KEY (userId, movieId)
        )
    """)

    conn.commit()
    conn.close()
    print("Database schema initialized successfully.")

def import_csv_to_table(csv_path, table_name):
    """
    Imports a CSV file (or directory of part-files from Spark) into a SQLite table.
    """
    engine = get_engine()
    
    # If csv_path is a directory (Spark format), find the actual CSV file inside
    if os.path.isdir(csv_path):
        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV part-files found in directory {csv_path}")
        # Take the first non-empty CSV part-file
        csv_path = os.path.join(csv_path, csv_files[0])

    print(f"Importing {csv_path} into table '{table_name}'...")
    
    # Load in chunks to prevent memory issues for larger files
    chunksize = 100000
    first_chunk = True
    
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if first_chunk:
            chunk.to_sql(table_name, con=engine, if_exists="replace", index=False)
            first_chunk = False
        else:
            chunk.to_sql(table_name, con=engine, if_exists="append", index=False)

    print(f"Table '{table_name}' imported successfully.")

def get_db_stats():
    """Returns basic counts and statistics about the cached database."""
    if not os.path.exists(DB_PATH):
        return {"movies": 0, "ratings": 0, "similarities": 0, "recommendations": 0}
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {}
    for table in ["movies", "ratings", "similarities", "recommendations"]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            stats[table] = 0
            
    conn.close()
    return stats

def get_unique_users():
    """Returns a list of unique User IDs present in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT userId FROM recommendations ORDER BY userId")
        users = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        users = []
    conn.close()
    return users

def get_user_history(user_id, limit=10):
    """Retrieves a user's top-rated movies from their history."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT r.movieId, m.title, r.rating 
        FROM ratings r
        JOIN movies m ON r.movieId = m.movieId
        WHERE r.userId = ?
        ORDER BY r.rating DESC, r.timestamp DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(user_id, limit))
    conn.close()
    return df

def get_user_recommendations(user_id, limit=10):
    """Retrieves cached recommendation scores (ALS, Content, Hybrid) for a user."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT rec.movieId, m.title, rec.als_score, rec.content_score, rec.hybrid_score
        FROM recommendations rec
        JOIN movies m ON rec.movieId = m.movieId
        WHERE rec.userId = ?
        ORDER BY rec.hybrid_score DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(user_id, limit))
    conn.close()
    return df

def get_similar_movies(movie_id, limit=5):
    """Retrieves similar movies for a given movie based on TF-IDF content similarities."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT s.similarMovieId as movieId, m.title, s.similarity
        FROM similarities s
        JOIN movies m ON s.similarMovieId = m.movieId
        WHERE s.movieId = ?
        ORDER BY s.similarity DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(movie_id, limit))
    conn.close()
    return df

def get_movies_by_genre(genre_keyword, limit=20):
    """Search for movies matching a specific genre keyword."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT movieId, title, text_features
        FROM movies
        WHERE text_features LIKE ?
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(f"%{genre_keyword}%", limit))
    conn.close()
    return df

def search_movies(search_query, limit=10):
    """Search for movies by title."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT movieId, title, text_features
        FROM movies
        WHERE title LIKE ?
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(f"%{search_query}%", limit))
    conn.close()
    return df
