"""
Natural Language Processing (NLP) Recommendation Engine
Implements:
1. Text Preprocessing (Tokenization, Lowercasing, Stopword Removal, Stemming)
2. Content-based Metadata Soup aggregation (Overview + Genres + Keywords + Director + Cast)
3. TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer using NumPy
4. Cosine Similarity Engine with Multi-Selection User Profile aggregation
5. Feature Explainability (Extracts top shared NLP terms and semantic weights)
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Comprehensive standard English stopwords list
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as',
    'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot',
    'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d',
    'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s',
    'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
    'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
    'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
    'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves', 'also', 'one', 'two', 'along', 'whose', 'upon'
}

def simple_stem(word: str) -> str:
    """Lightweight suffix-stripping stemmer for NLP token normalization."""
    word = word.lower()
    if word.endswith('sses'):
        return word[:-2]
    elif word.endswith('ies'):
        return word[:-2]
    elif word.endswith('ss'):
        return word
    elif word.endswith('s') and len(word) > 3:
        return word[:-1]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('ly') and len(word) > 4:
        return word[:-2]
    return word

def tokenize(text: str) -> List[str]:
    """Tokenize text into cleaned, stemmed, non-stopword tokens."""
    if not text:
        return []
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    tokens = []
    for w in words:
        if w not in STOPWORDS:
            tokens.append(simple_stem(w))
    return tokens


class TFIDFRecommender:
    """
    TF-IDF Vector Space Model & Cosine Similarity Engine.
    Engineered for fast, interpretable movie recommendation.
    """
    def __init__(self, movies: List[Dict[str, Any]]):
        self.movies = movies
        self.movie_map = {m['id']: m for m in movies}
        self.title_to_id = {m['title'].strip().lower(): m['id'] for m in movies}
        
        # Build NLP metadata soup for each movie
        self.soups: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.vocabulary: Dict[str, int] = {}
        self.inv_vocabulary: Dict[int, str] = {}
        self.idf: np.ndarray = np.array([])
        self.tfidf_matrix: np.ndarray = np.array([])
        
        self._fit()

    def _build_soup(self, movie: Dict[str, Any]) -> str:
        """
        Creates a weighted metadata string combining:
        - Plot overview (primary semantic context)
        - Genres (repeated for higher keyword weight)
        - Keywords & Tags (thematic concepts)
        - Director & Cast names (director/star style matching)
        """
        overview = movie.get('overview', '')
        genres = movie.get('genres', '').replace('|', ' ')
        keywords = " ".join(movie.get('keywords', []))
        director = movie.get('director', '')
        cast = " ".join(movie.get('cast', []))
        
        # Weighted soup: genres and keywords repeated to emphasize thematic match
        soup = f"{overview} {genres} {genres} {keywords} {keywords} {director} {director} {cast}"
        return soup

    def _fit(self):
        """Calculates TF-IDF representations for all movies in the catalog."""
        self.soups = [self._build_soup(m) for m in self.movies]
        self.doc_tokens = [tokenize(soup) for soup in self.soups]
        
        # Compute Document Frequency (DF)
        doc_count = len(self.doc_tokens)
        df_counter = Counter()
        for tokens in self.doc_tokens:
            unique_in_doc = set(tokens)
            df_counter.update(unique_in_doc)
            
        # Build vocabulary (filter out terms that appear in < 1 doc)
        sorted_terms = sorted([term for term, count in df_counter.items() if count >= 1])
        self.vocabulary = {term: idx for idx, term in enumerate(sorted_terms)}
        self.inv_vocabulary = {idx: term for term, idx in self.vocabulary.items()}
        vocab_size = len(self.vocabulary)
        
        # Calculate smooth IDF: ln((1 + N) / (1 + DF)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocabulary.items():
            df_val = df_counter[term]
            self.idf[idx] = math.log((1.0 + doc_count) / (1.0 + df_val)) + 1.0

        # Build TF-IDF Matrix (N_movies x Vocab_size)
        self.tfidf_matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    term_idx = self.vocabulary[token]
                    tf = count / float(total_tokens)
                    self.tfidf_matrix[i, term_idx] = tf * self.idf[term_idx]

        # L2-normalize each movie vector so cosine similarity is simply the dot product
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = self.tfidf_matrix / norms

    def get_user_vector(self, selected_ids: List[int]) -> np.ndarray:
        """
        Builds a composite user taste vector by taking the mean of selected movies'
        TF-IDF vectors and re-normalizing it to unit length.
        """
        if not selected_ids:
            return np.zeros(len(self.vocabulary), dtype=np.float32)
        
        id_to_idx = {m['id']: idx for idx, m in enumerate(self.movies)}
        indices = [id_to_idx[mid] for mid in selected_ids if mid in id_to_idx]
        if not indices:
            return np.zeros(len(self.vocabulary), dtype=np.float32)
            
        sub_vectors = self.tfidf_matrix[indices]
        mean_vec = np.mean(sub_vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec

    def explain_similarity(self, source_movie_ids: List[int], target_movie_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explains why target_movie was recommended:
        Extracts the highest-contributing shared TF-IDF terms between the user selection and target movie.
        """
        user_vec = self.get_user_vector(source_movie_ids)
        target_idx = next((i for i, m in enumerate(self.movies) if m['id'] == target_movie_id), None)
        if target_idx is None:
            return []
            
        target_vec = self.tfidf_matrix[target_idx]
        element_product = user_vec * target_vec
        top_indices = np.argsort(element_product)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            score = float(element_product[idx])
            if score > 0.0001:
                term = self.inv_vocabulary.get(int(idx), "")
                explanations.append({
                    "term": term,
                    "importance": round(score * 100, 2)
                })
        return explanations

    def recommend(
        self,
        selected_ids: List[int],
        genre_filter: Optional[List[str]] = None,
        min_rating: float = 0.0,
        year_min: int = 1900,
        year_max: int = 2030,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Finds the top N most similar movies to the user's selected movies.
        Calculates 'Rating (for you)' matching the Streamlit visual reference (e.g. 90-93%).
        """
        if not selected_ids:
            # If no selection, return top rated movies
            candidates = sorted(self.movies, key=lambda x: x.get('avg_rating', 0), reverse=True)
            results = []
            for i, m in enumerate(candidates[:top_n], start=1):
                res = dict(m)
                res['rank'] = i
                res['score_for_you'] = m.get('avg_rating', 80)
                res['similarity_raw'] = 0.8
                res['shared_keywords'] = [t['term'] for t in self.explain_similarity([], m['id'])]
                results.append(res)
            return results

        user_vec = self.get_user_vector(selected_ids)
        
        # Calculate Cosine Similarities via Dot Product
        sim_scores = np.dot(self.tfidf_matrix, user_vec)
        
        # Max-pool similarity against individual selections to boost tight matches
        id_to_idx = {m['id']: idx for idx, m in enumerate(self.movies)}
        selected_indices = [id_to_idx[mid] for mid in selected_ids if mid in id_to_idx]
        max_sims = np.zeros(len(self.movies), dtype=np.float32)
        if selected_indices:
            pairwise = np.dot(self.tfidf_matrix, self.tfidf_matrix[selected_indices].T)
            max_sims = np.max(pairwise, axis=1)

        # Blend composite similarity (70%) + peak pairwise similarity (30%)
        blended_sim = (0.7 * sim_scores) + (0.3 * max_sims)

        selected_set = set(selected_ids)
        ranked_indices = np.argsort(blended_sim)[::-1]
        
        recommendations = []
        rank_counter = 1

        for idx in ranked_indices:
            movie = self.movies[idx]
            m_id = movie['id']
            
            # Do not recommend the exact movies user has already selected
            if m_id in selected_set:
                continue
                
            # Filter by Genre if specified
            if genre_filter:
                movie_genres = [g.strip().lower() for g in movie.get('genres', '').split('|')]
                if not any(gf.lower() in movie_genres for gf in genre_filter):
                    continue
                    
            # Filter by Release Year
            year = movie.get('year', 2000)
            if year < year_min or year > year_max:
                continue
                
            # Filter by Minimum Average Rating
            avg_rating = movie.get('avg_rating', 0)
            if avg_rating < min_rating:
                continue

            raw_sim = float(blended_sim[idx])
            
            # Scale similarity to match the Streamlit "Rating (for you)" reference:
            # Baseline is ~88%, scaling smoothly up to 94-96% for top NLP matches
            scaled_rating = int(round(88 + (raw_sim * 12)))
            scaled_rating = max(75, min(96, scaled_rating))
            
            shared_terms = self.explain_similarity(selected_ids, m_id, top_k=4)

            rec_item = dict(movie)
            rec_item['rank'] = rank_counter
            rec_item['score_for_you'] = scaled_rating
            rec_item['similarity_raw'] = round(raw_sim, 4)
            rec_item['top_shared_terms'] = shared_terms
            rec_item['shared_keywords'] = [item['term'] for item in shared_terms]
            
            recommendations.append(rec_item)
            rank_counter += 1
            if len(recommendations) >= top_n:
                break

        return recommendations
