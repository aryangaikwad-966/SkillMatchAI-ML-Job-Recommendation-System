"""
Job Recommendation System - ML Module (Kaggle Dataset Version)
==============================================================

A production-quality machine learning module for recommending jobs based on user skills.
Adapted to work with the Kaggle "AI-Powered Job Recommendations Dataset" (50,000 jobs).

Features:
- Kaggle dataset loading and adaptation
- Text preprocessing and cleaning
- TF-IDF vectorization
- Cosine similarity-based recommendations
- Model persistence using joblib
- Skill matching and explanation

Author: ML Engineer
Date: 2026-04-12
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
from .kaggle_adapter import KaggleDatasetAdapter


class JobRecommendationSystem:
    """
    A machine learning-based job recommendation system.
    
    Uses TF-IDF vectorization and cosine similarity to recommend
    the most relevant jobs based on user skills and preferences.
    Supports both small sample datasets and large Kaggle datasets.
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialize the Job Recommendation System.
        
        Args:
            model_dir (str): Directory to store trained models
        """
        self.model_dir = model_dir
        self.vectorizer = None
        self.job_vectors = None
        self.jobs_data = None
        self.jobs_corpus = None
        self.adapter = KaggleDatasetAdapter()
        
        # Create model directory if it doesn't exist
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
    
    # ==================== DATA LOADING ====================
    def load_kaggle_dataset(self, file_path, sample=None):
        """
        Load and adapt Kaggle job recommendations dataset.
        
        Args:
            file_path (str): Path to Kaggle CSV file
            sample (int): Optional - use random sample of N rows for faster processing
            
        Returns:
            pd.DataFrame: Adapted dataset with standardized columns
        """
        # Use adapter to load and standardize
        self.jobs_data = self.adapter.load_kaggle_dataset(file_path, sample=sample)
        
        # Create corpus
        self.jobs_corpus = self.adapter.create_corpus(self.jobs_data)
        
        return self.jobs_data
    
    def load_data(self, file_path):
        """
        Load job dataset from CSV file.
        Supports both sample datasets and Kaggle dataset.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        # Load CSV file
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # Check if it's Kaggle format (has 'Job Title' column with capitals)
        if 'Job Title' in df.columns or 'job title' in df.columns.str.lower().tolist():
            print("Detected Kaggle dataset format - using adapter...")
            return self.load_kaggle_dataset(file_path)
        
        # Otherwise, assume it's the sample format with standardized columns
        print("Detected sample dataset format")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Validate required columns
        required_columns = ['job_title', 'required_skills']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Handle missing values
        df['required_skills'] = df['required_skills'].fillna('')
        df['job_title'] = df['job_title'].fillna('Unknown')
        
        # Add optional columns if present
        if 'industry' not in df.columns:
            df['industry'] = ''
        if 'location' not in df.columns:
            df['location'] = ''
        if 'experience_level' not in df.columns:
            df['experience_level'] = ''
        if 'company' not in df.columns:
            df['company'] = ''
        
        # Store data
        self.jobs_data = df.reset_index(drop=True)
        
        # Create corpus
        self.jobs_corpus = self.adapter.create_corpus(self.jobs_data)
        
        print(f"✓ Loaded {len(self.jobs_data)} jobs from dataset")
        return self.jobs_data
    
    # ==================== TEXT PREPROCESSING ====================
    @staticmethod
    def preprocess_text(text):
        """
        Clean and preprocess text data.
        
        Steps:
        1. Convert to lowercase
        2. Remove special characters and punctuation
        3. Remove extra whitespace
        4. Preserve words (only remove non-alphanumeric except spaces)
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # ==================== VECTORIZATION ====================
    def vectorize_data(self, min_df=1, max_df=0.95, max_features=3000):
        """
        Apply TF-IDF vectorization to the corpus.
        
        Args:
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency (0-1 scale)
            max_features (int): Maximum number of features
            
        Returns:
            scipy.sparse matrix: Document-term matrix
        """
        print(f"\nApplying TF-IDF vectorization...")
        print(f"  - min_df: {min_df}")
        print(f"  - max_df: {max_df}")
        print(f"  - max_features: {max_features}")
        print(f"  - ngram_range: (1, 2) [unigrams + bigrams]")
        
        # Preprocess corpus
        processed_corpus = [self.preprocess_text(doc) for doc in self.jobs_corpus]
        
        # Create vectorizer with bigram support for multi-word skills
        # e.g. "machine learning", "data analysis", "supply chain"
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Fit and transform
        self.job_vectors = self.vectorizer.fit_transform(processed_corpus)
        
        print(f"✓ Vectorized {self.job_vectors.shape[0]} documents")
        print(f"✓ Created {self.job_vectors.shape[1]}-dimensional feature space")
        
        return self.job_vectors
    
    # ==================== RECOMMENDATION ENGINE ====================
    def recommend_jobs(self, user_input, top_k=5, explain=True):
        """
        Recommend top-K jobs based on user input (skills).
        
        Args:
            user_input (str): User input describing desired job/skills
            top_k (int): Number of recommendations to return
            explain (bool): If True, include matched skills
            
        Returns:
            list: List of dicts with recommendations
                {
                    'job_id': int,
                    'job_title': str,
                    'company': str,
                    'location': str,
                    'similarity_score': float,
                    'matched_skills': list (if explain=True),
                    'skill_overlap_pct': float (if explain=True)
                }
        """
        if self.vectorizer is None or self.job_vectors is None:
            raise ValueError("System not trained. Call vectorize_data() first.")
        
        # Input validation
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty. Provide comma-separated skills.")
        
        # Clamp top_k to dataset size
        top_k = min(top_k, len(self.jobs_data))
        
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Transform user input using same vectorizer
        user_vector = self.vectorizer.transform([processed_input])
        
        # Compute cosine similarity with all jobs
        similarities = cosine_similarity(user_vector, self.job_vectors)[0]
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Warn if best score is very low
        best_score = float(similarities[top_indices[0]])
        if best_score < 0.05:
            print(f"⚠️  Warning: Best match score is very low ({best_score:.4f}).")
            print(f"   Your skills may not be well represented in the dataset.")
            print(f"   Try more specific or different skill terms.")
        
        # Parse user skills by comma (preserving multi-word skills)
        user_skills_set = set(s.strip().lower() for s in user_input.split(',') if s.strip())
        
        # Build recommendations list
        recommendations = []
        for idx in top_indices:
            job_info = {
                'job_id': int(idx),
                'job_title': self.jobs_data.loc[idx, 'job_title'],
                'company': self.jobs_data.loc[idx, 'company'] if 'company' in self.jobs_data.columns else 'N/A',
                'location': self.jobs_data.loc[idx, 'location'] if 'location' in self.jobs_data.columns else 'N/A',
                'industry': self.jobs_data.loc[idx, 'industry'] if 'industry' in self.jobs_data.columns else 'N/A',
                'experience_level': self.jobs_data.loc[idx, 'experience_level'] if 'experience_level' in self.jobs_data.columns else 'N/A',
                'similarity_score': float(similarities[idx]),
            }
            
            # Add matched skills if requested
            if explain:
                job_skills = self.jobs_data.loc[idx, 'required_skills']
                # Split job skills by comma (same format as user input)
                job_skills_set = set(s.strip().lower() for s in str(job_skills).split(',') if s.strip())
                
                # Find exact matches (multi-word aware)
                matched = user_skills_set & job_skills_set
                
                # Also find partial/substring matches for flexibility
                # e.g. user says "python" and job has "python programming"
                for user_skill in user_skills_set:
                    for job_skill in job_skills_set:
                        if user_skill in job_skill or job_skill in user_skill:
                            matched.add(job_skill)
                
                job_info['matched_skills'] = sorted(list(matched)) if matched else []
                
                # Skill overlap percentage
                if user_skills_set:
                    job_info['skill_overlap_pct'] = len(matched) / len(user_skills_set) * 100
                else:
                    job_info['skill_overlap_pct'] = 0.0
            
            recommendations.append(job_info)
        
        return recommendations
    
    # ==================== MODEL PERSISTENCE ====================
    def save_model(self, name='job_recommender'):
        """
        Save trained model to disk using joblib.
        
        Args:
            name (str): Model name (without extension)
        """
        vectorizer_path = os.path.join(self.model_dir, f'{name}_vectorizer.pkl')
        data_path = os.path.join(self.model_dir, f'{name}_data.pkl')
        
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump({
            'jobs_data': self.jobs_data,
            'jobs_corpus': self.jobs_corpus,
            'job_vectors': self.job_vectors
        }, data_path)
        
        print(f"✓ Model saved to {vectorizer_path}")
        print(f"✓ Data saved to {data_path}")
    
    def load_model(self, name='job_recommender'):
        """
        Load trained model from disk.
        
        Args:
            name (str): Model name (without extension)
        """
        vectorizer_path = os.path.join(self.model_dir, f'{name}_vectorizer.pkl')
        data_path = os.path.join(self.model_dir, f'{name}_data.pkl')
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model not found: {vectorizer_path}")
        
        self.vectorizer = joblib.load(vectorizer_path)
        data = joblib.load(data_path)
        
        self.jobs_data = data['jobs_data']
        self.jobs_corpus = data['jobs_corpus']
        self.job_vectors = data['job_vectors']
        
        print(f"✓ Model loaded from {vectorizer_path}")
    
    # ==================== TRAINING PIPELINE ====================
    def train(self, data_path, sample=None):
        """
        Complete training pipeline.
        
        Args:
            data_path (str): Path to dataset CSV
            sample (int): Optional - use random sample of N rows
            
        Returns:
            dict: Training summary
        """
        print("="*70)
        print("TRAINING JOB RECOMMENDATION SYSTEM")
        print("="*70)
        
        # Load data
        print("\n[1/3] Loading dataset...")
        self.load_data(data_path)
        
        # Vectorize
        print("\n[2/3] Vectorizing corpus...")
        self.vectorize_data()
        
        # Save model
        print("\n[3/3] Saving model...")
        self.save_model()
        
        summary = {
            'total_jobs': len(self.jobs_data),
            'vectorizer_features': self.job_vectors.shape[1],
            'total_documents': self.job_vectors.shape[0],
            'model_size_mb': os.path.getsize(os.path.join(self.model_dir, 'job_recommender_vectorizer.pkl')) / (1024*1024)
        }
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"  • Jobs in dataset: {summary['total_jobs']}")
        print(f"  • Features created: {summary['vectorizer_features']}")
        print(f"  • Model size: {summary['model_size_mb']:.2f} MB")
        print("="*70)
        
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_skill_stats(jobs_data):
    """Extract and display skill statistics."""
    return KaggleDatasetAdapter.extract_skill_stats(jobs_data)


def display_recommendations(recommendations, show_matched_skills=True):
    """Display recommendations in formatted table."""
    print("\n" + "="*100)
    print("TOP JOB RECOMMENDATIONS")
    print("="*100)
    
    for rank, rec in enumerate(recommendations, 1):
        overlap_str = f" | Skill Match: {rec['skill_overlap_pct']:.0f}%" if 'skill_overlap_pct' in rec else ""
        print(f"\n[{rank}] {rec['job_title']} | Score: {rec['similarity_score']:.3f}{overlap_str}")
        print(f"    Company: {rec.get('company', 'N/A')}")
        print(f"    Location: {rec.get('location', 'N/A')}")
        print(f"    Industry: {rec.get('industry', 'N/A')}")
        print(f"    Level: {rec.get('experience_level', 'N/A')}")
        
        if show_matched_skills and rec.get('matched_skills'):
            print(f"    ✓ Matched Skills: {', '.join(rec['matched_skills'])}")
        elif show_matched_skills:
            print(f"    ✗ No direct skill keyword matches (similarity is based on TF-IDF context)")
    
    print("\n" + "="*100)
