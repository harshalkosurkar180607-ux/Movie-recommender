# 🎬 Movie Recommendation System using Natural Language Processing (NLP) & Python

> **College Mini Project**  
> Developed by **Harshal**  
> A full-stack, content-based movie recommendation system replicating modern Streamlit (`movie-rex.streamlit.app`) and Netflix-style cinema user interfaces powered by a Python NLP engine.

---

## 🌟 Key Features

1. **Streamlit UI Layout (Image 1 Reference)**:
   - **Multi-Select Input**: Dynamic tag input with chips (e.g. `Band of Brothers (2001) ✕`, `The Godfather: Part II (1974) ✕`) and autocomplete search across the entire indexed movie catalog.
   - **Collapsible Filtering Options**: Interactive accordion to filter by genres, release year range, minimum average rating, and recommendation count.
   - **Action Button**: Primary `Find similar Movies 🎬` button with smooth hover glow and ripple effects.
   - **Dark Mode Results Table**:
     - `#` Rank
     - `Title` (with mini poster thumbnail)
     - `Rating (for you)` (Coral red progress bars with exact percentage matches)
     - `Average Rating (all users)`
     - `Genres` (Pipe-separated format e.g. `Crime|Drama`, `Action|Crime|Drama|IMAX`)

2. **Netflix Cinema Poster Grid (Image 2 Reference)**:
   - High-definition movie posters and backdrops.
   - 3D card hover scale, match rating badges, and release year tags.
   - Interactive modal popup with full plot synopsis, cast, director, and NLP keyword tags.

3. **Natural Language Processing (NLP) Vector Inspector**:
   - Live breakdown of **TF-IDF weights** and **Cosine Similarity dot products**.
   - Displays exact matching NLP terms (e.g., `corleone`, `mafia`, `organized crime`, `brotherhood`, `war`) explaining *why* each movie was recommended.

---

## 🧠 NLP & Machine Learning Architecture

```
Raw Movie Data (Plot Overview, Genres, Keywords, Director, Cast)
                             ↓
              [1. Text Preprocessing & Cleaning]
          (Lowercasing, Regex Tokenization, Stopwords Removal)
                             ↓
              [2. Stemming / Morphological Normalization]
                   (Suffix stripping, e.g., 'crimes' → 'crime')
                             ↓
              [3. Weighted Metadata Soup Construction]
         (Plot + 2x Genres + 2x Keywords + 1.5x Director + Cast)
                             ↓
              [4. TF-IDF Vectorization Matrix]
        TF(t, d) = count(t, d) / total_terms(d)
        IDF(t) = ln[(1 + N) / (1 + DF(t))] + 1
        TF-IDF(t, d) = TF(t, d) × IDF(t)
                             ↓
              [5. L2 Normalization (Unit Sphere)]
                     ||v||_2 = sqrt(Σ v_i^2)
                             ↓
              [6. Cosine Similarity & Profile Aggregation]
            cos(θ) = (A · B) / (||A|| × ||B||)
                             ↓
              [7. Ranking & Scaled Recommendation Score]
```

---

## 📂 Project Structure

```
mini project/
├── data/
│   └── movies_dataset.py     # 65+ curated movies with posters, overviews & keywords
├── static/
│   ├── index.html            # Streamlit & Netflix frontend template
│   ├── style.css             # Dark cinema UI styling & coral progress bars
│   └── app.js                # Frontend controller & API integration
├── nlp_engine.py             # Python NLP pipeline (TF-IDF & Cosine Similarity)
├── main.py                   # FastAPI server & REST API endpoints
├── run.bat                   # Double-click launcher for Windows
└── README.md                 # Complete project documentation
```

---

## 🚀 How to Run the Project

### Option A: Double-Click Launcher (Windows)
Simply double-click [`run.bat`](file:///c:/Users/HP/OneDrive/Desktop/Harshal%20Cllg/mini%20project/run.bat). It will:
1. Start the Python FastAPI server on `http://127.0.0.1:8000`.
2. Automatically open your default web browser to the application.

### Option B: Terminal Command
Open PowerShell or Command Prompt in this folder and run:
```bash
python main.py
```
Then navigate to: **`http://127.0.0.1:8000`** in your browser.

---

## 🎓 College Presentation & Viva Questions

### Q1: What type of recommendation system is this?
> **Answer:** It is a **Content-Based Recommendation System** powered by **Natural Language Processing (NLP)**. Instead of relying solely on collaborative user ratings, it analyzes the textual content (plot synopsis, thematic tags, genres, directors, and actors) to calculate semantic similarity.

### Q2: Why use TF-IDF instead of simple word counts?
> **Answer:** Simple word count (Bag of Words) treats frequent words as more important, even though common words carry less discriminating information. **TF-IDF (Term Frequency - Inverse Document Frequency)** penalizes terms that appear across many movies while emphasizing rare, distinctive keywords that specifically define a movie's plot or genre.

### Q3: What is the formula for Cosine Similarity?
> **Answer:** 
> $$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$
> Since our TF-IDF vectors are already $L_2$-normalized to unit length ($\|\mathbf{A}\|_2 = 1$), the cosine similarity is simplified to the dot product $\mathbf{A} \cdot \mathbf{B}$.

### Q4: How does the system handle multiple selected movies?
> **Answer:** When multiple movies are selected (e.g. *Band of Brothers* and *The Godfather*), the engine computes a composite user profile vector (the centroid of the selected movies' unit vectors) and performs blended similarity scoring to capture both overall taste and individual affinities.
