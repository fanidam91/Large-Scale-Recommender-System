import os
import random
import csv
import pandas as pd
from database import init_db, import_csv_to_table

def generate_mock_data():
    """
    Generates synthetic MovieLens-like CSV exports (simulating Spark output)
    and loads them into the local SQLite database for offline dashboard testing.
    """
    print("Generating synthetic MovieLens dataset to simulate Spark exports...")
    
    # Create export directory structure
    export_dir = os.path.join("data", "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    paths = {
        "movies": os.path.join(export_dir, "movies_metadata"),
        "ratings": os.path.join(export_dir, "ratings_history"),
        "similarities": os.path.join(export_dir, "similarities"),
        "recommendations": os.path.join(export_dir, "recommendations")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    # 1. Generate Movies Metadata
    movies = [
        (1, "Toy Story (1995)", "Adventure Animation Children Comedy Fantasy classic Pixar"),
        (2, "Jumanji (1995)", "Adventure Children Fantasy board-game family"),
        (3, "Heat (1995)", "Action Crime Thriller robbery police Al-Pacino"),
        (4, "Sabrina (1995)", "Comedy Romance remake Audrey-Hepburn"),
        (5, "GoldenEye (1995)", "Action Adventure Thriller James-Bond spy"),
        (6, "Sense and Sensibility (1995)", "Drama Romance Jane-Austen classic"),
        (7, "Get Shorty (1995)", "Comedy Crime Thriller hollywood gangster"),
        (8, "Braveheart (1995)", "Action Drama War historical scotland"),
        (9, "Clueless (1995)", "Comedy Romance high-school fashion"),
        (10, "Twelve Monkeys (1995)", "Mystery Sci-Fi Thriller time-travel dystopia"),
        (11, "Apollo 13 (1995)", "Drama IMAX space NASA true-story"),
        (12, "Babe (1995)", "Children Drama talking-pig farm cute"),
        (13, "Casino (1995)", "Crime Drama las-vegas mafia martin-scorsese"),
        (14, "Richard III (1995)", "Drama War shakespeare"),
        (15, "Seven (1995)", "Mystery Thriller serial-killer dark"),
        (16, "Usual Suspects, The (1995)", "Crime Mystery Thriller twist classic"),
        (17, "Clerks (1994)", "Comedy independent black-and-white cult"),
        (18, "Shawshank Redemption, The (1994)", "Crime Drama prison hope classic-top"),
        (19, "Pulp Fiction (1994)", "Comedy Crime Drama Quentin-Tarantino cult"),
        (20, "Forrest Gump (1994)", "Comedy Drama Romance War history classic"),
        (21, "Star Wars: Episode IV - A New Hope (1977)", "Action Adventure Sci-Fi space-opera classic cult"),
        (22, "Matrix, The (1999)", "Action Sci-Fi Thriller cyberpunk virtual-reality"),
        (23, "Blade Runner (1982)", "Action Sci-Fi Thriller cyberpunk noir classic"),
        (24, "Alien (1979)", "Horror Sci-Fi space creature-feature classic"),
        (25, "Terminator 2: Judgment Day (1991)", "Action Sci-Fi cyborg time-travel action-packed"),
    ]
    
    movies_df = pd.DataFrame(movies, columns=["movieId", "title", "text_features"])
    movies_file = os.path.join(paths["movies"], "part-00000.csv")
    movies_df.to_csv(movies_file, index=False)
    print(f"Generated {len(movies)} mock movies in {movies_file}")

    # 2. Generate Ratings History
    ratings = []
    # Create ratings for 10 users
    random.seed(42)
    genres_preference = {
        1: ["Sci-Fi", "Action"],       # User 1 likes Action/Sci-Fi
        2: ["Children", "Adventure"],  # User 2 likes Kids movies
        3: ["Drama", "Romance"],       # User 3 likes Drama/Romance
        4: ["Crime", "Thriller"],      # User 4 likes Crime/Thriller
        5: ["Comedy", "Romance"],      # User 5 likes Comedy/Romance
    }
    
    for user_id in range(1, 11):
        pref = genres_preference.get(user_id, ["Action", "Comedy", "Drama", "Sci-Fi"])
        for movie_id, title, tags in movies:
            # Check if movie matches user preferences
            has_pref = any(p in tags for p in pref)
            
            # Probability of rating
            prob = 0.85 if has_pref else 0.2
            if random.random() < prob:
                # Rating score
                rating_score = round(random.uniform(3.5, 5.0), 1) if has_pref else round(random.uniform(1.0, 3.0), 1)
                # Timestamp
                timestamp = 840823000 + random.randint(1000, 1000000)
                ratings.append((user_id, movie_id, rating_score, timestamp))

    ratings_df = pd.DataFrame(ratings, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_file = os.path.join(paths["ratings"], "part-00000.csv")
    ratings_df.to_csv(ratings_file, index=False)
    print(f"Generated {len(ratings)} mock ratings in {ratings_file}")

    # 3. Generate Movie Similarities (TF-IDF Cosine Similarities)
    similarities = []
    for m1_idx, m1 in enumerate(movies):
        for m2_idx, m2 in enumerate(movies):
            if m1[0] == m2[0]:
                continue
                
            # Compute intersection of genre tags
            tags1 = set(m1[2].split())
            tags2 = set(m2[2].split())
            intersection = tags1.intersection(tags2)
            union = tags1.union(tags2)
            
            # Calculate Jaccard similarity as mock cosine similarity
            sim = len(intersection) / len(union) if union else 0.0
            if sim > 0.15:
                # Add random noise
                sim = min(1.0, max(0.0, sim + random.uniform(-0.05, 0.05)))
                similarities.append((m1[0], m2[0], round(sim, 4)))

    similarities_df = pd.DataFrame(similarities, columns=["movieId", "similarMovieId", "similarity"])
    similarities_file = os.path.join(paths["similarities"], "part-00000.csv")
    similarities_df.to_csv(similarities_file, index=False)
    print(f"Generated {len(similarities)} mock similarities in {similarities_file}")

    # 4. Generate Recommendations (ALS + Content + Hybrid)
    recommendations = []
    for user_id in range(1, 11):
        # We recommend movies the user hasn't rated yet, or rated highly in mock predictions
        user_rated = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())
        pref = genres_preference.get(user_id, ["Action", "Comedy", "Drama", "Sci-Fi"])
        
        candidates = []
        for movie_id, title, tags in movies:
            has_pref = any(p in tags for p in pref)
            
            # Generate mock recommendation scores
            als_score = round(random.uniform(3.5, 4.8), 2) if has_pref else round(random.uniform(1.5, 3.2), 2)
            content_score = round(random.uniform(3.2, 4.5), 2) if has_pref else round(random.uniform(1.0, 2.5), 2)
            hybrid_score = round(0.6 * als_score + 0.4 * content_score, 2)
            
            candidates.append((user_id, movie_id, als_score, content_score, hybrid_score))
            
        # Sort and take top 15 recommendations per user
        candidates.sort(key=lambda x: x[4], reverse=True)
        recommendations.extend(candidates[:15])

    recs_df = pd.DataFrame(recommendations, columns=["userId", "movieId", "als_score", "content_score", "hybrid_score"])
    recs_file = os.path.join(paths["recommendations"], "part-00000.csv")
    recs_df.to_csv(recs_file, index=False)
    print(f"Generated {len(recommendations)} mock recommendations in {recs_file}")

    # Initialize SQLite database and import generated datasets
    print("\nInitializing database schema...")
    init_db()
    
    print("Importing generated CSV files into database tables...")
    import_csv_to_table(paths["movies"], "movies")
    import_csv_to_table(paths["ratings"], "ratings")
    import_csv_to_table(paths["similarities"], "similarities")
    import_csv_to_table(paths["recommendations"], "recommendations")
    
    print("\nMock setup completed! Local SQLite database loaded and ready for dashboard usage.")

if __name__ == "__main__":
    generate_mock_data()
