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

        self.df.columns = [
            c.strip().lower().replace(" ", "_")
            for c in self.df.columns
        ]

        rename_map = {
            "approx_cost(for_two_people)": "approx_cost",
            "listed_in(type)": "type",
            "listed_in(city)": "city",
        }
        self.df.rename(columns=rename_map, inplace=True)

        # Clean ratings
        if "rate" in self.df.columns:
            self.df["rate"] = (
                self.df["rate"]
                .astype(str)
                .str.replace("/5", "", regex=False)
                .str.strip()
            )
            self.df["rate"] = pd.to_numeric(
                self.df["rate"],
                errors="coerce"
            )
            self.df["rate"] = self.df["rate"].fillna(
                self.df["rate"].mean()
            )

        # Clean cost column
        if "approx_cost" in self.df.columns:
            self.df["approx_cost"] = (
                self.df["approx_cost"]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            self.df["approx_cost"] = pd.to_numeric(
                self.df["approx_cost"],
                errors="coerce"
            )
            self.df["approx_cost"] = self.df["approx_cost"].fillna(
                self.df["approx_cost"].median()
            )

        # Fill missing values
        for col in ["cuisines", "location", "rest_type", "name"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("Unknown")

        # Clean votes
        if "votes" in self.df.columns:
            self.df["votes"] = pd.to_numeric(
                self.df["votes"],
                errors="coerce"
            ).fillna(0)

        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df

    def build_model(self):
        """Build TF-IDF model using restaurant features"""

        if self.df is None:
            self.load_data()

        feature_cols = [
            c for c in ["name", "cuisines", "rest_type", "location"]
            if c in self.df.columns
        ]

        self.df["features"] = self.df[feature_cols].astype(str).agg(
            " ".join,
            axis=1
        )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["features"]
        )

    def recommend(
        self,
        restaurant_name=None,
        max_cost=None,
        min_rating=0.0,
        top_n=10
    ):
        """
        Hybrid recommendation based on:
        1. Restaurant name similarity
        2. Ratings
        3. Popularity (votes)
        4. Budget filter

        Parameters:
        ----------
        restaurant_name : str
            Restaurant name or keyword
        max_cost : float
            Maximum cost for two people
        min_rating : float
            Minimum rating
        top_n : int
            Number of recommendations

        Returns:
        -------
        DataFrame
        """

        if self.tfidf_matrix is None:
            self.build_model()

        df = self.df.copy()

        # Rating filter
        if "rate" in df.columns:
            df = df[df["rate"] >= min_rating]

        # Cost filter
        if max_cost is not None and "approx_cost" in df.columns:
            df = df[df["approx_cost"] <= max_cost]

        if df.empty:
            return pd.DataFrame()

        # Name-based similarity search
        query = restaurant_name if restaurant_name else ""

        if query:
            query_vec = self.vectorizer.transform([query])

            sim_scores = cosine_similarity(
                query_vec,
                self.tfidf_matrix[df.index]
            ).flatten()
        else:
            sim_scores = np.ones(len(df))

        df["similarity"] = sim_scores

        # Popularity score
        max_votes = (
            df["votes"].max()
            if "votes" in df.columns and df["votes"].max() > 0
            else 1
        )

        df["popularity_score"] = 0.0

        if "rate" in df.columns:
            df["popularity_score"] += (
                (df["rate"] / 5.0) * 0.7
            )

        if "votes" in df.columns:
            df["popularity_score"] += (
                np.log1p(df["votes"]) /
                np.log1p(max_votes)
            ) * 0.3

        # Hybrid score
        df["hybrid_score"] = (
            df["similarity"] * 0.6 +
            df["popularity_score"] * 0.4
        )

        # Bonus if restaurant name matches
        if restaurant_name and "name" in df.columns:
            mask = df["name"].str.contains(
                restaurant_name,
                case=False,
                na=False
            )

            if mask.any():
                df.loc[mask, "hybrid_score"] += 0.4

        result_cols = [
            c for c in [
                "name",
                "cuisines",
                "location",
                "rest_type",
                "rate",
                "votes",
                "approx_cost",
                "hybrid_score"
            ]
            if c in df.columns
        ]

        result = (
            df.sort_values(
                "hybrid_score",
                ascending=False
            )
            .head(top_n)
        )

        return result[result_cols].reset_index(drop=True)


if __name__ == "__main__":

    print("=" * 60)
    print("RESTAURANT RECOMMENDATION APP")
    print("Recommend Restaurants by Name, Budget & Rating")
    print("=" * 60)

    recommender = RestaurantRecommender("Zomato-data-.csv")

    recommender.load_data()

    print(f"\nLoaded {len(recommender.df)} restaurants")

    print("\nEnter your preferences:\n")

    restaurant_name = input(
        "Enter restaurant name (or keyword): "
    ).strip()

    max_cost = float(
        input(
            "Enter maximum cost for 2 people (₹): "
        )
    )

    rating = float(
        input(
            "Enter minimum rating (0-5): "
        )
    )

    top_n = int(
        input(
            "How many recommendations do you want? "
        )
    )

    recs = recommender.recommend(
        restaurant_name=restaurant_name,
        max_cost=max_cost,
        min_rating=rating,
        top_n=top_n
    )

    if recs.empty:
        print("\n❌ No matching restaurants found.")
    else:
        print(f"\n✅ Top {len(recs)} Recommendations:\n")
        print("=" * 80)

        for i, row in recs.iterrows():

            print(f"\n{i + 1}. {row['name']}")
            print(f"   Cuisine: {row.get('cuisines', 'N/A')}")
            print(f"   Location: {row.get('location', 'N/A')}")
            print(f"   Type: {row.get('rest_type', 'N/A')}")
            print(f"   Rating: {row.get('rate', 'N/A')}/5 ⭐")
            print(f"   Votes: {row.get('votes', 'N/A')}")
            print(f"   Cost (for 2): ₹{row.get('approx_cost', 'N/A')}")
            print(
                f"   Recommendation Score: "
                f"{row.get('hybrid_score', 0):.3f}"
            )

        print("\n" + "=" * 80)
