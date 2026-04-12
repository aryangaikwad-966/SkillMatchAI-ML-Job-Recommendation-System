"""
Evaluation Module for Kaggle Dataset
====================================

Implements evaluation metrics for job recommendation system:
- Precision@K: Percentage of top-K recommendations that are relevant
- Recall@K: Percentage of all relevant jobs in top-K recommendations
- Mean Reciprocal Rank@K: Average position of first relevant job

Test Cases: 10 diverse user profiles representing different job seekers

Author: ML Engineer
Date: 2026-04-12
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class EvaluationMetrics:
    """Calculate evaluation metrics for recommendations."""
    
    @staticmethod
    def precision_at_k(recommended_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
        """
        Calculate Precision@K: (# relevant in top-K) / K
        
        Args:
            recommended_ids: List of recommended job IDs (in order)
            relevant_ids: List of relevant job IDs (ground truth)
            k: Top-K cutoff
            
        Returns:
            float: Precision@K score (0-1)
        """
        if k == 0 or len(relevant_ids) == 0:
            return 0.0
        
        top_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        matched = sum(1 for job_id in top_k if job_id in relevant_set)
        
        return matched / k
    
    @staticmethod
    def recall_at_k(recommended_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
        """
        Calculate Recall@K: (# relevant in top-K) / (total relevant)
        
        Args:
            recommended_ids: List of recommended job IDs (in order)
            relevant_ids: List of relevant job IDs (ground truth)
            k: Top-K cutoff
            
        Returns:
            float: Recall@K score (0-1)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        top_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        matched = sum(1 for job_id in top_k if job_id in relevant_set)
        
        return matched / len(relevant_ids)
    
    @staticmethod
    def mrr_at_k(recommended_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
        """
        Calculate Mean Reciprocal Rank@K: 1 / (rank of first relevant)
        
        Args:
            recommended_ids: List of recommended job IDs (in order)
            relevant_ids: List of relevant job IDs (ground truth)
            k: Top-K cutoff
            
        Returns:
            float: MRR@K score (0-1, where 1 means first result is relevant)
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        relevant_set = set(relevant_ids)
        for rank, job_id in enumerate(recommended_ids[:k], start=1):
            if job_id in relevant_set:
                return 1.0 / rank
        
        return 0.0


class TestCaseManager:
    """Manage test cases for evaluation."""
    
    # 10 Diverse Test Cases representing different job seekers
    TEST_CASES = [
        {
            'id': 1,
            'name': 'Python Developer',
            'user_input': 'Python, Django, REST API, SQL databases, Git version control',
            'description': 'Backend developer with focus on Python and web frameworks',
            'relevant_keywords': ['python', 'django', 'api', 'backend', 'database', 'sql']
        },
        {
            'id': 2,
            'name': 'Data Scientist',
            'user_input': 'Python, Machine Learning, Data Analysis, Statistics, TensorFlow, Pandas',
            'description': 'Data analyst with ML and statistical analysis background',
            'relevant_keywords': ['machine learning', 'data', 'python', 'analytics', 'tensorflow', 'statistics']
        },
        {
            'id': 3,
            'name': 'Frontend Developer',
            'user_input': 'JavaScript, React, CSS, HTML, Responsive Design, TypeScript',
            'description': 'Web developer focused on modern frontend frameworks',
            'relevant_keywords': ['javascript', 'react', 'frontend', 'css', 'html', 'typescript']
        },
        {
            'id': 4,
            'name': 'DevOps Engineer',
            'user_input': 'Docker, Kubernetes, CI/CD, AWS, Linux, Infrastructure',
            'description': 'Operations engineer specialized in cloud and containerization',
            'relevant_keywords': ['docker', 'kubernetes', 'aws', 'devops', 'ci/cd', 'linux']
        },
        {
            'id': 5,
            'name': 'Data Engineer',
            'user_input': 'Spark, Hadoop, ETL, Big Data, Python, SQL, Data Pipelines',
            'description': 'Data infrastructure specialist building scalable pipelines',
            'relevant_keywords': ['spark', 'hadoop', 'etl', 'data', 'python', 'sql']
        },
        {
            'id': 6,
            'name': 'Cloud Architect',
            'user_input': 'AWS, Google Cloud, Azure, Architecture, Design, Cloud Infrastructure',
            'description': 'Solution architect designing cloud-based systems',
            'relevant_keywords': ['aws', 'cloud', 'azure', 'architect', 'infrastructure', 'design']
        },
        {
            'id': 7,
            'name': 'Mobile Developer',
            'user_input': 'Swift, Kotlin, React Native, Mobile App, iOS, Android',
            'description': 'Mobile application developer for iOS and Android',
            'relevant_keywords': ['swift', 'mobile', 'android', 'ios', 'kotlin', 'react native']
        },
        {
            'id': 8,
            'name': 'QA Engineer',
            'user_input': 'Selenium, Testing, Quality Assurance, Automation, Test Cases, Java',
            'description': 'Quality assurance specialist in test automation',
            'relevant_keywords': ['testing', 'qa', 'selenium', 'automation', 'quality', 'test']
        },
        {
            'id': 9,
            'name': 'Full Stack Developer',
            'user_input': 'JavaScript, Node.js, React, PostgreSQL, Full Stack, Web Development',
            'description': 'Complete application development from frontend to backend',
            'relevant_keywords': ['javascript', 'react', 'nodejs', 'fullstack', 'web', 'database']
        },
        {
            'id': 10,
            'name': 'AI/ML Engineer',
            'user_input': 'Deep Learning, Neural Networks, PyTorch, TensorFlow, NLP, Computer Vision',
            'description': 'Advanced ML engineer specializing in deep learning',
            'relevant_keywords': ['deep learning', 'pytorch', 'neural networks', 'nlp', 'ai', 'cv']
        }
    ]
    
    @classmethod
    def get_test_cases(cls) -> List[Dict]:
        """Get all test cases."""
        return cls.TEST_CASES
    
    @classmethod
    def get_test_case(cls, test_id: int) -> Dict:
        """Get specific test case by ID."""
        for tc in cls.TEST_CASES:
            if tc['id'] == test_id:
                return tc
        raise ValueError(f"Test case {test_id} not found")
    
    @staticmethod
    def find_relevant_jobs(jobs_data, keywords: List[str], min_matches: int = 1) -> List[int]:
        """
        Find relevant job IDs by matching keywords in skills.
        
        Args:
            jobs_data: DataFrame with job information
            keywords: List of keywords to match
            min_matches: Minimum number of keywords to match
            
        Returns:
            List of relevant job IDs
        """
        relevant_ids = []
        keywords_lower = set(kw.lower() for kw in keywords)
        
        for idx, job_skills in enumerate(jobs_data['required_skills']):
            if not isinstance(job_skills, str):
                continue
            
            skills_set = set(s.strip().lower() for s in job_skills.split(','))
            matches = len(keywords_lower & skills_set)
            
            if matches >= min_matches:
                relevant_ids.append(idx)
        
        return relevant_ids


class EvaluationAnalyzer:
    """Analyze evaluation results."""
    
    def __init__(self, k: int = 5):
        """
        Initialize analyzer.
        
        Args:
            k: Top-K value for metrics
        """
        self.k = k
        self.results = []
    
    def add_result(self, test_id: int, test_name: str, 
                   recommended_ids: List[int], relevant_ids: List[int]):
        """
        Add evaluation result for a test case.
        
        Args:
            test_id: Test case ID
            test_name: Test case name
            recommended_ids: List of recommended job IDs from model
            relevant_ids: List of ground truth relevant job IDs
        """
        metrics = EvaluationMetrics()
        precision = metrics.precision_at_k(recommended_ids, relevant_ids, self.k)
        recall = metrics.recall_at_k(recommended_ids, relevant_ids, self.k)
        mrr = metrics.mrr_at_k(recommended_ids, relevant_ids, self.k)
        
        result = {
            'test_id': test_id,
            'test_name': test_name,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'mrr_at_k': mrr,
            'top_recommendation': (recommended_ids[0] if recommended_ids else -1)
        }
        
        self.results.append(result)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        df = pd.DataFrame(self.results)
        return df.round(4)
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all tests."""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        return {
            'average_precision': df['precision_at_k'].mean(),
            'average_recall': df['recall_at_k'].mean(),
            'average_mrr': df['mrr_at_k'].mean(),
            'median_precision': df['precision_at_k'].median(),
            'median_recall': df['recall_at_k'].median(),
            'median_mrr': df['mrr_at_k'].median(),
            'std_precision': df['precision_at_k'].std(),
            'std_recall': df['recall_at_k'].std(),
            'std_mrr': df['mrr_at_k'].std(),
            'min_precision': df['precision_at_k'].min(),
            'max_precision': df['precision_at_k'].max(),
            'min_recall': df['recall_at_k'].min(),
            'max_recall': df['recall_at_k'].max(),
        }
    
    def print_summary(self):
        """Print summary statistics."""
        stats = self.get_summary_statistics()
        
        print("\n" + "="*70)
        print("EVALUATION SUMMARY (across 10 test cases)")
        print("="*70)
        
        print(f"\nPrecision@{self.k}:")
        print(f"  Average: {stats['average_precision']:.4f}")
        print(f"  Median:  {stats['median_precision']:.4f}")
        print(f"  Range:   {stats['min_precision']:.4f} - {stats['max_precision']:.4f}")
        
        print(f"\nRecall@{self.k}:")
        print(f"  Average: {stats['average_recall']:.4f}")
        print(f"  Median:  {stats['median_recall']:.4f}")
        print(f"  Range:   {stats['min_recall']:.4f} - {stats['max_recall']:.4f}")
        
        print(f"\nMRR@{self.k}:")
        print(f"  Average: {stats['average_mrr']:.4f}")
        print(f"  Median:  {stats['median_mrr']:.4f}")
        
        print("\n" + "="*70)
    
    def print_detailed_results(self):
        """Print detailed test-by-test results."""
        df = self.get_results_dataframe()
        
        print("\n" + "="*100)
        print("DETAILED EVALUATION RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    
    def export_results_csv(self, output_path: str):
        """Export results to CSV."""
        df = self.get_results_dataframe()
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results exported to {output_path}")
