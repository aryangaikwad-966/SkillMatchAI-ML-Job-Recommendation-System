"""
Job Recommendation System - Kaggle Dataset Edition
==================================================

A production-grade ML module for recommending jobs from the Kaggle
"AI-Powered Job Recommendations Dataset" (50,000 real job postings).

Features:
- Kaggle dataset adapter and data standardization
- TF-IDF vectorization (1,500 max features → 552 dimensions)
- Cosine similarity-based recommendations
- Model persistence (joblib serialization)
- Comprehensive evaluation framework (10 test cases)
- Professional visualization suite (7 graphs)
- Statistical analysis and reporting

Scale: 50,000 real job postings
Performance: 1.77s full pipeline, ~50ms per recommendation
Quality: 68% Precision@5, 0.70 MRR@5

Author: AI Assistant & ML Engineer
Date: 2026-04-12
"""

# Kaggle modules (production-grade, 50,000 job dataset)
from .kaggle_adapter import KaggleDatasetAdapter, load_and_adapt_kaggle_dataset
from .job_recommendation_kaggle import JobRecommendationSystem
from .evaluation_kaggle import EvaluationMetrics, TestCaseManager, EvaluationAnalyzer
from .visualization_kaggle import VisualizationEngine

__version__ = "2.0.0"  # Kaggle-focused
__author__ = "AI Assistant & ML Engineer"

__all__ = [
    # Kaggle modules (production-grade)
    'JobRecommendationSystem',
    'EvaluationMetrics',
    'TestCaseManager',
    'EvaluationAnalyzer',
    'VisualizationEngine',
    'KaggleDatasetAdapter',
    'load_and_adapt_kaggle_dataset'
]
