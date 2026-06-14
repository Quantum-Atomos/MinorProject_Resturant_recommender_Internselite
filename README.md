[README.md](https://github.com/user-attachments/files/28923217/README.md)
Name: Manimoy Karmakar
Roll number:AIML-A6/May-10007

# Restaurant Recommendation App

A hybrid restaurant recommendation system that suggests restaurants based on cuisine preference, location, and ratings using collaborative + content-based filtering.

## Features
- Cuisine-based filtering
- Location-based filtering (distance calculation)
- Rating-based ranking
- Hybrid recommendation score (content-based + popularity)
- Zomato-style API integration (mock/sample data included)

## Tech Stack
- Python 3
- Pandas, NumPy
- Scikit-learn (cosine similarity for content-based filtering)
- Requests (for Zomato/3rd-party API calls)

## Project Structure
```
restaurant_recommender/
│
├──resturant_recommender.py                # Main application
├── data/
│   └── Zomato-data-.csv(ffrom kaggle.com)  # Sample restaurant dataset
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```

When prompted, enter:
- Preferred cuisine (e.g., "Italian", "Chinese")
- Your location (latitude, longitude)
- Minimum rating threshold

The app will return a ranked list of recommended restaurants.

## How It Works
1. **Content-Based Filtering**: Computes similarity between cuisines using TF-IDF + cosine similarity.
2. **Location Filtering**: Calculates distance using the Haversine formula and filters nearby restaurants.
3. **Popularity/Rating Score**: Restaurants are ranked by average rating and number of votes.
4. **Hybrid Score**: Combines similarity score, rating score, and proximity score with weighted averaging.

## Zomato API Integration
To use real data, sign up for the Zomato API (or an alternative like Yelp/Google Places, since Zomato's public API has limited availability) and add your API key to `config.py`:
```python
ZOMATO_API_KEY = "your_api_key_here"
```

## Sample Output
```
Top Recommendations for 'Italian' cuisine near (22.57, 88.36):
1. La Pizzeria       - Rating: 4.5  Distance: 1.2 km
2. Pasta House       - Rating: 4.3  Distance: 2.5 km
3. Mama Mia          - Rating: 4.1  Distance: 0.8 km
```

## Future Improvements
- User-based collaborative filtering using ratings history
- Real-time integration with multiple food delivery APIs
- Web/mobile interface (Flask/React)
