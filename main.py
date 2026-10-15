"""
Movie Recommendation System - Web Server (FastAPI + Python NLP Engine)
Provides REST API endpoints and serves modern Streamlit-style frontend.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.movies_dataset import MOVIES
from nlp_engine import TFIDFRecommender

app = FastAPI(
    title="Movie Finder NLP Recommendation System",
    description="Natural Language Processing (TF-IDF & Cosine Similarity) Movie Recommender",
    version="1.0.0"
)

# Initialize NLP Recommender with Movie Catalog
recommender = TFIDFRecommender(MOVIES)

# Request Models
class RecommendRequest(BaseModel):
    movie_ids: List[int]
    genres: Optional[List[str]] = []
    min_rating: Optional[float] = 0.0
    year_min: Optional[int] = 1900
    year_max: Optional[int] = 2030
    top_n: Optional[int] = 10

@app.get("/api/movies")
def get_movies(query: Optional[str] = None):
    """Returns list of movies with optional title search for multi-select dropdown."""
    if query:
        q = query.lower().strip()
        matched = [
            m for m in MOVIES
            if q in m['title'].lower() or any(q in g.lower() for g in m['genres'].split('|'))
        ]
        return matched[:30]
    return MOVIES

@app.get("/api/genres")
def get_genres():
    """Extracts all unique genres available in the database."""
    unique_genres = set()
    for m in MOVIES:
        for g in m.get("genres", "").split("|"):
            if g.strip():
                unique_genres.add(g.strip())
    return sorted(list(unique_genres))

@app.post("/api/recommend")
def recommend_movies(req: RecommendRequest):
    """
    Computes top recommendations using TF-IDF Vector Space Model
    and Cosine Similarity across plot overviews, genres, keywords, and cast.
    """
    results = recommender.recommend(
        selected_ids=req.movie_ids,
        genre_filter=req.genres if req.genres else None,
        min_rating=req.min_rating or 0.0,
        year_min=req.year_min or 1900,
        year_max=req.year_max or 2030,
        top_n=req.top_n or 10
    )
    return {
        "status": "success",
        "count": len(results),
        "selected_ids": req.movie_ids,
        "recommendations": results
    }

@app.get("/api/movie/{movie_id}")
def get_movie_detail(movie_id: int):
    """Returns complete details of a movie."""
    movie = recommender.movie_map.get(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

@app.get("/api/explain")
def explain_nlp(selected_id: int, target_id: int):
    """
    Provides mathematical breakdown of NLP TF-IDF and Cosine Similarity:
    returns shared terms, TF-IDF importance, and similarity formula steps.
    """
    source_movie = recommender.movie_map.get(selected_id)
    target_movie = recommender.movie_map.get(target_id)
    if not source_movie or not target_movie:
        raise HTTPException(status_code=404, detail="Movie not found")
        
    shared_terms = recommender.explain_similarity([selected_id], target_id, top_k=8)
    
    return {
        "source": source_movie['title'],
        "target": target_movie['title'],
        "shared_terms": shared_terms,
        "source_overview": source_movie['overview'],
        "target_overview": target_movie['overview'],
        "formula": "Cosine Similarity: cos(θ) = (A · B) / (||A|| * ||B||)"
    }

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(">> Movie Finder - NLP Recommendation System Running!")
    print(">> Open in browser: http://127.0.0.1:8000")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
