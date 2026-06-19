import os
import sys

def verify_setup():
    print("HelixRec Verification Script")
    print("============================")

    # 1. Check imports
    print("\n1. Verifying imports...")
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sqlalchemy
        print("[OK] All required packages are successfully installed!")
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # 2. Run mock generator to seed database
    print("\n2. Seeding database with synthetic mock data...")
    try:
        from generate_mock_data import generate_mock_data
        generate_mock_data()
        print("[OK] Database successfully seeded!")
    except Exception as e:
        print(f"[ERROR] Database seeding failed: {e}")
        sys.exit(1)

    # 3. Test database queries
    print("\n3. Testing database queries...")
    try:
        import database as db
        
        users = db.get_unique_users()
        print(f"   Found {len(users)} unique user profiles in cache: {users}")
        if not users:
            raise ValueError("No user profiles returned.")
            
        test_user = users[0]
        
        history = db.get_user_history(test_user, limit=5)
        print(f"   User {test_user} rated history count: {len(history)}")
        if history.empty:
            raise ValueError("Empty user history returned.")
            
        recs = db.get_user_recommendations(test_user, limit=5)
        print(f"   User {test_user} recommendations count: {len(recs)}")
        if recs.empty:
            raise ValueError("Empty user recommendations returned.")
            
        similar = db.get_similar_movies(movie_id=1, limit=5)
        print(f"   Movie 1 similarities count: {len(similar)}")
        if similar.empty:
            raise ValueError("Empty similarities returned.")

        print("[OK] All database queries verified successfully!")
    except Exception as e:
        print(f"[ERROR] Database query testing failed: {e}")
        sys.exit(1)

    print("\n============================")
    print("SETUP VERIFICATION SUCCESSFUL!")
    print("To run the Streamlit dashboard: streamlit run src/app.py")

if __name__ == "__main__":
    verify_setup()
