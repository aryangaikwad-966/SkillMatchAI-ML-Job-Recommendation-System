"""
Kaggle Dataset Adapter Module
==============================

Converts the Kaggle "AI-Powered Job Recommendations Dataset" to the 
internal format used by the recommendation system.

Dataset: AI-Powered Job Recommendations Dataset
Columns: Job Title, Company, Location, Experience Level, Salary, Industry, Required Skills
Records: 50,000 job postings

Author: ML Engineer
Date: 2026-04-12
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class KaggleDatasetAdapter:
    """Adapter to transform Kaggle dataset into recommendation system format."""
    
    @staticmethod
    def load_kaggle_dataset(file_path, sample=None):
        """
        Load and adapt Kaggle dataset.
        
        Args:
            file_path (str): Path to Kaggle CSV file
            sample (int): If specified, return random sample of N rows
            
        Returns:
            pd.DataFrame: Adapted dataset with columns:
                - job_title: Job role
                - required_skills: Comma-separated skills
                - industry: Industry sector
                - location: Job location
                - experience_level: Career level
                - company: Hiring company
        """
        print(f"Loading Kaggle dataset from {file_path}...")
        
        # Load CSV
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Standardize column names (handle different casings)
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"\nStandardized columns: {list(df.columns)}")
        
        # Map Kaggle columns to internal format
        adapted_df = pd.DataFrame()
        
        # Job Title - required
        if 'job title' in df.columns:
            adapted_df['job_title'] = df['job title'].fillna('Unknown Job')
        else:
            raise ValueError("'Job Title' column required in dataset")
        
        # Required Skills - required (may be called 'skills' or 'required skills')
        if 'required skills' in df.columns:
            adapted_df['required_skills'] = df['required skills'].fillna('')
        elif 'skills' in df.columns:
            adapted_df['required_skills'] = df['skills'].fillna('')
        else:
            raise ValueError("'Required Skills' column required in dataset")
        
        # Industry - optional but useful for corpus
        adapted_df['industry'] = df['industry'].fillna('') if 'industry' in df.columns else ''
        
        # Location - optional
        adapted_df['location'] = df['location'].fillna('') if 'location' in df.columns else ''
        
        # Experience Level - optional
        adapted_df['experience_level'] = df['experience level'].fillna('') if 'experience level' in df.columns else ''
        
        # Company - optional
        adapted_df['company'] = df['company'].fillna('') if 'company' in df.columns else ''
        
        # Salary - optional (useful for analysis)
        if 'salary' in df.columns:
            adapted_df['salary'] = df['salary']
        else:
            adapted_df['salary'] = np.nan
        
        # Remove rows with no job title or skills
        initial_count = len(adapted_df)
        adapted_df = adapted_df[
            (adapted_df['job_title'].notna()) &
            (adapted_df['job_title'] != '') &
            (adapted_df['required_skills'].notna()) &
            (adapted_df['required_skills'] != '')
        ].reset_index(drop=True)
        
        print(f"\nRemoved {initial_count - len(adapted_df)} rows with missing data")
        print(f"Final dataset shape: {adapted_df.shape}")
        print(f"\nSample rows:")
        print(adapted_df.head(3))
        
        # Optionally sample for faster processing
        if sample and sample < len(adapted_df):
            adapted_df = adapted_df.sample(n=sample, random_state=42).reset_index(drop=True)
            print(f"\nSampled to {len(adapted_df)} rows")
        
        return adapted_df
    
    @staticmethod
    def create_corpus(df):
        """
        Create text corpus by combining relevant job fields.
        
        Args:
            df (pd.DataFrame): Adapted dataset
            
        Returns:
            pd.Series: Combined text corpus for each job
        """
        print("\nCreating text corpus...")
        
        # Combine fields: job_title + industry + required_skills
        # Weight is implicit: all fields treated equally
        corpus = (
            df['job_title'].astype(str).str.lower() + ' ' +
            df['industry'].astype(str).str.lower() + ' ' +
            df['required_skills'].astype(str).str.lower()
        )
        
        # Clean up extra spaces
        corpus = corpus.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        print(f"Corpus created: {len(corpus)} documents")
        print(f"\nSample corpus entries:")
        for i in range(min(3, len(corpus))):
            preview = corpus.iloc[i][:100] + "..."
            print(f"  [{i}] {preview}")
        
        return corpus
    
    @staticmethod
    def extract_skill_stats(df):
        """
        Extract insights from skills data.
        
        Args:
            df (pd.DataFrame): Adapted dataset
            
        Returns:
            dict: Skill statistics
        """
        print("\nExtracting skill statistics...")
        
        # Split skills and count frequency
        all_skills = []
        for skills_str in df['required_skills'].dropna():
            if isinstance(skills_str, str) and skills_str.strip():
                skills = [s.strip() for s in skills_str.split(',')]
                all_skills.extend(skills)
        
        # Count skill frequency
        from collections import Counter
        skill_counts = Counter(all_skills)
        top_skills = skill_counts.most_common(20)
        
        print(f"\nTotal unique skills: {len(skill_counts)}")
        print(f"Top 10 most common skills:")
        for skill, count in top_skills[:10]:
            print(f"  • {skill}: {count} jobs")
        
        return {
            'total_unique_skills': len(skill_counts),
            'top_skills': top_skills,
            'skill_counts': skill_counts
        }
    
    @staticmethod
    def extract_experience_stats(df):
        """
        Extract insights from experience level data.
        
        Args:
            df (pd.DataFrame): Adapted dataset
            
        Returns:
            dict: Experience level statistics
        """
        if 'experience_level' not in df.columns or df['experience_level'].empty:
            return {}
        
        exp_counts = df['experience_level'].value_counts()
        print(f"\nExperience levels in dataset:")
        for exp_level, count in exp_counts.items():
            if exp_level and str(exp_level).strip():
                print(f"  • {exp_level}: {count} jobs")
        
        return exp_counts.to_dict()
    
    @staticmethod
    def save_adapted_dataset(df, output_path):
        """
        Save adapted dataset locally.
        
        Args:
            df (pd.DataFrame): Adapted dataset
            output_path (str): Path to save CSV
        """
        df.to_csv(output_path, index=False)
        print(f"\n✓ Adapted dataset saved to {output_path}")


# ============================================================================
# CONVENIENT FUNCTION FOR QUICK LOADING
# ============================================================================

def load_and_adapt_kaggle_dataset(kaggle_file_path, sample=None):
    """
    Convenience function to load and adapt Kaggle dataset in one call.
    
    Args:
        kaggle_file_path (str): Path to Kaggle CSV
        sample (int): Optional - return random sample of N rows
        
    Returns:
        tuple: (adapted_df, corpus, skill_stats, experience_stats)
    """
    # Load and adapt
    adapter = KaggleDatasetAdapter()
    df = adapter.load_kaggle_dataset(kaggle_file_path, sample=sample)
    
    # Create corpus
    corpus = adapter.create_corpus(df)
    
    # Extract statistics
    skill_stats = adapter.extract_skill_stats(df)
    experience_stats = adapter.extract_experience_stats(df)
    
    return df, corpus, skill_stats, experience_stats
