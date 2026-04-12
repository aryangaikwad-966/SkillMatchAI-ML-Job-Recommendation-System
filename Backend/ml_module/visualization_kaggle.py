"""
Visualization Module for Kaggle Dataset
=======================================

Creates publication-quality visualizations for evaluation results:
- Bar charts for top recommendations with scores
- Histograms for similarity score distribution
- Bar charts for most common skills in dataset

Author: ML Engineer
Date: 2026-04-12
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path


class VisualizationEngine:
    """Generate visualizations for job recommendation system."""
    
    def __init__(self, output_dir='graphs'):
        """
        Initialize visualization engine.
        
        Args:
            output_dir: Directory to save graphs
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_top_recommendations(self, recommendations: List[Dict], test_name: str) -> str:
        """
        Create bar chart of top job recommendations with scores.
        
        Args:
            recommendations: List of recommendation dicts with 'job_title' and 'similarity_score'
            test_name: Name of test case
            
        Returns:
            str: Path to saved graph
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        titles = [rec['job_title'][:50] for rec in recommendations]
        scores = [rec['similarity_score'] for rec in recommendations]
        
        # Create bar chart
        bars = ax.barh(range(len(titles)), scores, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
        
        # Customize
        ax.set_yticks(range(len(titles)))
        ax.set_yticklabels(titles)
        ax.set_xlabel('Similarity Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Top Job Recommendations for: {test_name}', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f'01_recommendations_{test_name.lower().replace(" ", "_")}.png'
        filepath = f'{self.output_dir}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_similarity_distribution(self, similarities: List[float]) -> str:
        """
        Create histogram of similarity score distribution.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            str: Path to saved graph
        """
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Create histogram
        n, bins, patches = ax.hist(similarities, bins=30, color='steelblue', 
                                    alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add statistical lines
        mean_score = np.mean(similarities)
        median_score = np.median(similarities)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2.5, 
                  label=f'Median: {median_score:.3f}')
        
        # Customize
        ax.set_xlabel('Similarity Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Similarity Scores\n(Across All Recommendations)', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics box
        stats_text = f'Total: {len(similarities)}\nMin: {min(similarities):.3f}\nMax: {max(similarities):.3f}\nStd Dev: {np.std(similarities):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        filepath = f'{self.output_dir}/02_similarity_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_metrics_comparison(self, results_df: pd.DataFrame) -> str:
        """
        Create comparison of Precision, Recall, and MRR across test cases.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            str: Path to saved graph
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        test_names = results_df['test_name'].values
        x_pos = np.arange(len(test_names))
        
        # Precision@5
        ax = axes[0]
        bars = ax.bar(x_pos, results_df['precision_at_k'], color='skyblue', alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        ax.axhline(results_df['precision_at_k'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Avg: {results_df["precision_at_k"].mean():.3f}')
        ax.set_ylabel('Precision@5', fontsize=10, fontweight='bold')
        ax.set_title('Precision@5', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([n.replace(' ', '\n') for n in test_names], fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Recall@5
        ax = axes[1]
        bars = ax.bar(x_pos, results_df['recall_at_k'], color='lightgreen', alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        ax.axhline(results_df['recall_at_k'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Avg: {results_df["recall_at_k"].mean():.3f}')
        ax.set_ylabel('Recall@5', fontsize=10, fontweight='bold')
        ax.set_title('Recall@5', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([n.replace(' ', '\n') for n in test_names], fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # MRR@5
        ax = axes[2]
        bars = ax.bar(x_pos, results_df['mrr_at_k'], color='salmon', alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        ax.axhline(results_df['mrr_at_k'].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'Avg: {results_df["mrr_at_k"].mean():.3f}')
        ax.set_ylabel('MRR@5', fontsize=10, fontweight='bold')
        ax.set_title('Mean Reciprocal Rank@5', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([n.replace(' ', '\n') for n in test_names], fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Evaluation Metrics Across 10 Test Cases', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        filepath = f'{self.output_dir}/03_metrics_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_average_metrics(self, results_df: pd.DataFrame) -> str:
        """
        Create summary bar chart of average metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            str: Path to saved graph
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Precision@5', 'Recall@5', 'MRR@5']
        avg_values = [
            results_df['precision_at_k'].mean(),
            results_df['recall_at_k'].mean(),
            results_df['mrr_at_k'].mean()
        ]
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        bars = ax.bar(metrics, avg_values, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5, width=0.6)
        
        # Add value labels
        for bar, value in zip(bars, avg_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Benchmark lines
        ax.axhline(0.5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Baseline (50%)')
        ax.axhline(0.7, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='Good (70%)')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title('Average Performance Metrics\n(10 Test Cases)', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = f'{self.output_dir}/04_average_metrics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_top_skills(self, skill_stats: Dict, top_n: int = 15) -> str:
        """
        Create bar chart of most common skills in dataset.
        
        Args:
            skill_stats: Dictionary with skill statistics from KaggleDatasetAdapter
            top_n: Number of top skills to display
            
        Returns:
            str: Path to saved graph
        """
        if 'top_skills' not in skill_stats or not skill_stats['top_skills']:
            print("No skill statistics available")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Extract top skills
        top_skills = skill_stats['top_skills'][:top_n]
        skills = [s[0] for s in top_skills]
        counts = [s[1] for s in top_skills]
        
        # Create bar chart
        bars = ax.barh(range(len(skills)), counts, color='teal', alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 5, bar.get_y() + bar.get_height()/2,
                   f'{count}', va='center', fontsize=9, fontweight='bold')
        
        # Customize
        ax.set_yticks(range(len(skills)))
        ax.set_yticklabels(skills)
        ax.set_xlabel('Number of Jobs', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Common Skills in Dataset', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = f'{self.output_dir}/05_top_skills.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_comprehensive_report(self, recommendations_by_test: Dict,
                                     similarities_all: List[float],
                                     results_df: pd.DataFrame,
                                     skill_stats: Dict) -> List[str]:
        """
        Generate all visualizations in one call.
        
        Args:
            recommendations_by_test: Dict of test_name -> recommendations
            similarities_all: All similarity scores from all tests
            results_df: DataFrame with evaluation results
            skill_stats: Skill statistics
            
        Returns:
            List of paths to generated graphs
        """
        print("\nGenerating visualizations...")
        
        filepaths = []
        
        # 1. Top recommendations for each test (sample 3)
        sample_tests = list(recommendations_by_test.items())[:3]
        for test_name, recs in sample_tests:
            filepath = self.plot_top_recommendations(recs, test_name)
            filepaths.append(filepath)
            print(f"  ✓ Generated: {filepath}")
        
        # 2. Similarity distribution
        filepath = self.plot_similarity_distribution(similarities_all)
        filepaths.append(filepath)
        print(f"  ✓ Generated: {filepath}")
        
        # 3. Metrics comparison
        filepath = self.plot_metrics_comparison(results_df)
        filepaths.append(filepath)
        print(f"  ✓ Generated: {filepath}")
        
        # 4. Average metrics
        filepath = self.plot_average_metrics(results_df)
        filepaths.append(filepath)
        print(f"  ✓ Generated: {filepath}")
        
        # 5. Top skills
        filepath = self.plot_top_skills(skill_stats, top_n=15)
        if filepath:
            filepaths.append(filepath)
            print(f"  ✓ Generated: {filepath}")
        
        print(f"\n✓ Generated {len(filepaths)} visualizations in '{self.output_dir}/'")
        
        return filepaths
