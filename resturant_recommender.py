import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RestaurantRecommender:
    def __init__(self, csv_path="Zomato-data-.csv"):
        self.csv_path = csv_path
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None

    def load_data(self):
        """Load and clean restaurant data from CSV file"""
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]

        rename_map = {
            "approx_cost(for_two_people)": "approx_cost",
            "listed_in(type)": "type",
            "listed_in(city)": "city",
        }
        self.df.rename(columns=rename_map, inplace=True)

        if "rate" in self.df.columns:
            self.df["rate"] = (self.df["rate"].astype(str).str.replace("/5", "", regex=False).str.strip())
            self.df["rate"] = pd.to_numeric(self.df["rate"], errors="coerce")
            self.df["rate"] = self.df["rate"].fillna(self.df["rate"].mean())

        if "approx_cost" in self.df.columns:
            self.df["approx_cost"] = (self.df["approx_cost"].astype(str).str.replace(",", "", regex=False))
            self.df["approx_cost"] = pd.to_numeric(self.df["approx_cost"], errors="coerce")
            self.df["approx_cost"] = self.df["approx_cost"].fillna(self.df["approx_cost"].median())

        for col in ["cuisines", "location", "rest_type", "name"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("Unknown")

        if "votes" in self.df.columns:
            self.df["votes"] = pd.to_numeric(self.df["votes"], errors="coerce").fillna(0)

        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def build_model(self):
        """Build TF-IDF model for content-based similarity"""
        if self.df is None:
            self.load_data()

        feature_cols = [c for c in ["cuisines", "rest_type", "location"] if c in self.df.columns]
        self.df["features"] = self.df[feature_cols].astype(str).agg(" ".join, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["features"])

    def recommend(self, cuisine=None, location=None, min_rating=0.0, top_n=10):
        """
        Hybrid recommendation combining content-based filtering + popularity boost

        Parameters:
        - cuisine: Preferred cuisine (e.g., 'Indian', 'Pizza')
        - location: Preferred location (e.g., 'Berhampore', 'Kolkata')
        - min_rating: Minimum rating threshold
        - top_n: Number of recommendations

        Returns: DataFrame with top N recommendations
        """
        if self.tfidf_matrix is None:
            self.build_model()

        df = self.df.copy()

        if "rate" in df.columns:
            df = df[df["rate"] >= min_rating]

        if df.empty:
            return pd.DataFrame()

        query_parts = []
        if cuisine:
            query_parts.append(cuisine)
        if location:
            query_parts.append(location)
        query = " ".join(query_parts) if query_parts else ""

        if query:
            query_vec = self.vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix[df.index]).flatten()
        else:
            sim_scores = np.ones(len(df))

        df["similarity"] = sim_scores

        max_votes = df["votes"].max() if "votes" in df.columns and df["votes"].max() > 0 else 1
        df["popularity_score"] = 0.0
        if "rate" in df.columns:
            df["popularity_score"] += (df["rate"] / 5.0) * 0.7
        if "votes" in df.columns:
            df["popularity_score"] += (np.log1p(df["votes"]) / np.log1p(max_votes)) * 0.3

        df["hybrid_score"] = (df["similarity"] * 0.6) + (df["popularity_score"] * 0.4)

        if cuisine and "cuisines" in df.columns:
            mask = df["cuisines"].str.contains(cuisine, case=False, na=False)
            if mask.any():
                df.loc[mask, "hybrid_score"] += 0.2

        if location and "location" in df.columns:
            mask = df["location"].str.contains(location, case=False, na=False)
            if mask.any():
                df.loc[mask, "hybrid_score"] += 0.2

        result_cols = [c for c in ["name", "cuisines", "location", "rest_type", "rate", "votes", "approx_cost", "hybrid_score"] if c in df.columns]
        result = df.sort_values("hybrid_score", ascending=False).head(top_n)
        return result[result_cols].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RESTAURANT RECOMMENDATION APP")
    print("Based on Cuisine, Location, and Ratings")
    print("=" * 60)

    recommender = RestaurantRecommender("Zomato-data-.csv")
    recommender.load_data()


    print(f"\nLoaded {len(recommender.df)} restaurants\n")

    cuisine = input("Enter cuisine (or blank for any): ").strip() or None
    location = input("Enter location (or blank for any): ").strip() or None
    rating_input = input("Enter min rating (e.g., 3.5, or blank for 0): ").strip()
    rating = float(rating_input) if rating_input else 0.0
    top_n = int(input("Number of recommendations (default 10): ").strip() or "10")

    recs = recommender.recommend(cuisine=cuisine, location=location, min_rating=rating, top_n=top_n)

    if recs.empty:
        print("\n❌ No matching restaurants found")
    else:
        print(f"\n✅ Top {len(recs)} Recommendations:\n")
        print("=" * 80)
        for i, row in recs.iterrows():
            print(f"\n{i+1}. {row['name']}")
            print(f"   Cuisine: {row.get('cuisines', 'N/A')}")
            print(f"   Location: {row.get('location', 'N/A')}")
            print(f"   Type: {row.get('rest_type', 'N/A')}")
            print(f"   Rating: {row.get('rate', 'N/A')}/5 ⭐")
            print(f"   Votes: {row.get('votes', 'N/A')}")
            print(f"   Cost (for 2): ₹{row.get('approx_cost', 'N/A')}")
            print(f"   Score: {row.get('hybrid_score', 0):.3f}")
        print("\n" + "=" * 80)