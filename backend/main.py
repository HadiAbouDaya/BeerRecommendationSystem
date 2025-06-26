from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Beer Recommendation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
style_encoder = None
brewery_encoder = None
beer_db = None

class UserPreferences(BaseModel):
    username: Optional[str] = "Anonymous User"
    beer_styles: List[str]
    abv_range: Optional[dict] = {"min": 0, "max": 100}
    breweries: Optional[List[str]] = []
    location: Optional[str] = "Not specified"

class BeerRecommendation(BaseModel):
    rank: int
    beer_name: str
    beer_style: str
    brewery_name: str
    abv: float
    predicted_rating: float
    avg_rating: float
    reason: str

@app.on_event("startup")
async def load_model():
    global model, style_encoder, brewery_encoder, beer_db
    try:
        logger.info("Loading model and encoders...")
        
        # Load TensorFlow model
        model = tf.saved_model.load("models/preference_recommender")
        
        # Load encoders
        style_encoder = joblib.load("models/style_encoder.pkl")
        brewery_encoder = joblib.load("models/brewery_encoder.pkl")
        
        # Load beer database
        beer_db = pd.read_csv("models/beer_database.csv")
        
        logger.info("Model and data loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/beer-styles")
async def get_beer_styles():
    """Get all available beer styles"""
    if style_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    styles = sorted(style_encoder.classes_.tolist())
    return {"beer_styles": styles}

@app.get("/breweries")
async def get_breweries():
    """Get all available breweries"""
    if beer_db is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    breweries = sorted(beer_db['brewery_name'].unique().tolist())
    return {"breweries": breweries}

@app.post("/recommend", response_model=List[BeerRecommendation])
async def recommend_beers(preferences: UserPreferences, top_n: int = 5):
    """Generate beer recommendations based on user preferences"""
    if model is None or beer_db is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        logger.info(f"Generating recommendations for {preferences.username}")
        
        # Filter beer database
        filtered_beers = beer_db.copy()
        
        # Filter by known encodings first
        known_styles = set(style_encoder.classes_)
        known_breweries = set(brewery_encoder.classes_)
        
        filtered_beers = filtered_beers[
            filtered_beers['beer_style'].isin(known_styles) & 
            filtered_beers['brewery_id'].isin(known_breweries)
        ]
        
        # Filter by user preferences
        if preferences.beer_styles:
            filtered_beers = filtered_beers[
                filtered_beers['beer_style'].isin(preferences.beer_styles)
            ]
        
        if preferences.abv_range:
            min_abv = preferences.abv_range.get('min', 0)
            max_abv = preferences.abv_range.get('max', 100)
            filtered_beers = filtered_beers[
                (filtered_beers['beer_abv'] >= min_abv) & 
                (filtered_beers['beer_abv'] <= max_abv)
            ]
        
        if preferences.breweries:
            filtered_beers = filtered_beers[
                filtered_beers['brewery_name'].isin(preferences.breweries)
            ]
        
        if len(filtered_beers) == 0:
            raise HTTPException(status_code=404, detail="No beers match your preferences")
        
        # Prepare features for prediction
        features = {
            "inputs": tf.convert_to_tensor(
                style_encoder.transform(filtered_beers['beer_style']), dtype=tf.float32
            ),
            "inputs_1": tf.convert_to_tensor(
                brewery_encoder.transform(filtered_beers['brewery_id']), dtype=tf.float32
            ),
            "inputs_2": tf.convert_to_tensor(
                filtered_beers['beer_abv'].values, dtype=tf.float32
            ),
            "inputs_3": tf.convert_to_tensor(
                filtered_beers['review_aroma'].values, dtype=tf.float32
            ),
            "inputs_4": tf.convert_to_tensor(
                filtered_beers['review_appearance'].values, dtype=tf.float32
            ),
        }
        
        # Get predictions
        predictions = model.signatures["serving_default"](**features)["output_0"].numpy().flatten()
        
        # Add predictions and sort
        filtered_beers = filtered_beers.copy()
        filtered_beers['predicted_rating'] = predictions
        top_recommendations = filtered_beers.sort_values(
            by='predicted_rating', ascending=False
        ).head(top_n)
        
        # Format results
        recommendations = []
        for i, (_, beer) in enumerate(top_recommendations.iterrows(), 1):
            recommendations.append(BeerRecommendation(
                rank=i,
                beer_name=beer['beer_name'],
                beer_style=beer['beer_style'],
                brewery_name=beer['brewery_name'],
                abv=round(beer['beer_abv'], 1),
                predicted_rating=round(beer['predicted_rating'], 2),
                avg_rating=round(beer['review_overall'], 2),
                reason=f"Matches your preference for {beer['beer_style']} beers"
            ))
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # ← This is critical for Docker
        port=8000,
        reload=False  # Disable reload in production/Docker
    )