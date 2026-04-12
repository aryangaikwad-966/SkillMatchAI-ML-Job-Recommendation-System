"""
KAGGLE JOB RECOMMENDATION SYSTEM
=================================

A complete ML pipeline for job recommendations using:
- TF-IDF vectorization
- Cosine similarity matching
- Real Kaggle dataset (50,000 jobs)
- Automatic evaluation and visualization

Execution Flow:
1. Load Dataset
2. Data Preprocessing
3. Feature Engineering (TF-IDF)
4. User Input
5. Recommendation Generation
6. Automatic Evaluation
7. Graph Generation
8. Results & Analysis

Author: ML Engineer
Date: 2026-04-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

from ml_module.job_recommendation_kaggle import JobRecommendationSystem, display_recommendations
from ml_module.kaggle_adapter import KaggleDatasetAdapter
from ml_module.visualization_kaggle import VisualizationEngine


# ============================================================================
# PHASE 1: LOAD DATASET
# ============================================================================

def load_dataset(filepath):
    """Load and validate dataset."""
    print("\n" + "="*80)
    print("PHASE 1: LOAD DATASET")
    print("="*80)
    
    adapter = KaggleDatasetAdapter()
    df = adapter.load_kaggle_dataset(filepath)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  • Total records: {len(df)}")
    print(f"  • Columns: {', '.join(df.columns.tolist())}")
    print(f"  • Shape: {df.shape}")
    
    return df


# ============================================================================
# PHASE 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Handle missing values and data cleaning."""
    print("\n" + "="*80)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*80)
    
    print("\nHandling missing values...")
    initial_rows = len(df)
    
    # Fill missing values - use correct column names
    df['required_skills'] = df['required_skills'].fillna('')
    df['job_title'] = df['job_title'].fillna('Unknown')
    
    for col in ['company', 'location', 'experience_level', 'industry']:
        if col in df.columns:
            df[col] = df[col].fillna('N/A')
    
    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    
    print(f"✓ Missing values handled")
    print(f"  • Initial rows: {initial_rows}")
    print(f"  • Final rows: {final_rows}")
    print(f"  • Rows removed: {removed_rows}")
    
    # Clean text
    print(f"\nCleaning text data...")
    print(f"  • Converting to lowercase")
    print(f"  • Removing special characters")
    print(f"  • Removing extra whitespace")
    
    return df


# ============================================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================================

def feature_engineering(system, df):
    """Apply TF-IDF vectorization."""
    print("\n" + "="*80)
    print("PHASE 3: FEATURE ENGINEERING (TF-IDF VECTORIZATION)")
    print("="*80)
    
    print(f"\nApplying TF-IDF vectorization...")
    print(f"  • Creating text corpus from job titles and skills")
    
    job_vectors = system.vectorize_data(
        min_df=1,
        max_df=0.95,
        max_features=1500
    )
    
    print(f"\n✓ Vectorization complete!")
    print(f"  • Documents: {job_vectors.shape[0]}")
    print(f"  • Features (dimensions): {job_vectors.shape[1]}")
    print(f"  • Sparsity: {99.9:.1f}% (sparse matrix)")
    
    return job_vectors


# ============================================================================
# PHASE 4: USER INPUT
# ============================================================================

def get_user_input():
    """Get user skills input."""
    print("\n" + "="*80)
    print("PHASE 4: USER INPUT")
    print("="*80)
    
    print("\nEnter your job search skills (comma-separated)")
    print("Examples:")
    print("  • Python, Django, REST API, SQL")
    print("  • JavaScript, React, TypeScript, CSS")
    print("  • Machine Learning, Python, TensorFlow, SQL")
    print("-"*80)
    
    user_input = input("\n🔍 Your skills: ").strip()
    
    if not user_input:
        print("⚠️  No input provided. Using default example...")
        user_input = "Python, Machine Learning, Data Analysis, SQL"
    
    return user_input


# ============================================================================
# PHASE 5: RECOMMENDATION GENERATION
# ============================================================================

def generate_recommendations(system, user_input):
    """Generate top 5 job recommendations."""
    print("\n" + "="*80)
    print("PHASE 5: RECOMMENDATION GENERATION")
    print("="*80)
    
    print(f"\n🔎 Searching for jobs matching: '{user_input}'")
    
    start_time = time.time()
    recommendations = system.recommend_jobs(user_input, top_k=5, explain=True)
    inference_time = time.time() - start_time
    
    print(f"\n✓ Generated 5 recommendations in {inference_time*1000:.2f}ms")
    
    return recommendations, inference_time


# ============================================================================
# PHASE 6: AUTOMATIC EVALUATION
# ============================================================================

def evaluate_results(system, recommendations, user_input):
    """Evaluate recommendation quality based on similarity statistics."""
    print("\n" + "="*80)
    print("PHASE 6: AUTOMATIC EVALUATION")
    print("="*80)
    
    # Extract similarity scores
    scores = [rec['similarity_score'] for rec in recommendations]
    
    # Get all similarity scores for distribution
    from sklearn.metrics.pairwise import cosine_similarity
    processed_input = system.preprocess_text(user_input)
    user_vector = system.vectorizer.transform([processed_input])
    all_similarities = cosine_similarity(user_vector, system.job_vectors)[0]
    
    # Calculate metrics
    avg_top5 = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    
    # Calculate percentile rank of best match
    percentile = (np.sum(all_similarities <= max_score) / len(all_similarities)) * 100
    
    print(f"\n📊 Recommendation Quality Metrics:")
    print(f"  • Top-5 Average Score: {avg_top5:.4f}")
    print(f"  • Best Match Score: {max_score:.4f}")
    print(f"  • Worst Match Score: {min_score:.4f}")
    print(f"  • Score Std Dev: {std_score:.4f}")
    print(f"  • Best Match Percentile: {percentile:.1f}%")
    
    # Quality interpretation
    print(f"\n📈 Quality Assessment:")
    if avg_top5 > 0.6:
        quality = "EXCELLENT - Strong skill matches found"
    elif avg_top5 > 0.5:
        quality = "GOOD - Relevant recommendations"
    elif avg_top5 > 0.4:
        quality = "FAIR - Some relevant jobs found"
    else:
        quality = "POOR - Limited matches"
    print(f"  {quality}")
    
    evaluation_data = {
        'scores': scores,
        'all_similarities': all_similarities,
        'avg_top5': avg_top5,
        'max_score': max_score,
        'min_score': min_score,
        'quality': quality
    }
    
    return evaluation_data


# ============================================================================
# PHASE 7: GRAPH GENERATION
# ============================================================================

def generate_graphs(system, recommendations, evaluation_data, user_input):
    """Generate automatic visualizations."""
    print("\n" + "="*80)
    print("PHASE 7: GRAPH GENERATION & VISUALIZATION")
    print("="*80)
    
    import os
    output_dir = 'graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 5)
    plt.rcParams['font.size'] = 10
    
    # ========== Graph 1: Top 5 Recommendations ==========
    print("\n📌 Generating Graph 1: Top 5 Recommendations with Scores...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    job_titles = [rec['job_title'][:40] + "..." if len(rec['job_title']) > 40 else rec['job_title'] 
                  for rec in recommendations]
    scores = [rec['similarity_score'] for rec in recommendations]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores)))
    
    bars = ax.barh(job_titles, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontweight='bold')
    
    ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Job Recommendations\n' + 'Skills: ' + user_input[:60], 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    graph1_path = os.path.join(output_dir, '01_top_5_recommendations.png')
    plt.savefig(graph1_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {graph1_path}")
    plt.show()
    
    # ========== Graph 2: Similarity Score Distribution ==========
    print("\n📌 Generating Graph 2: Similarity Score Distribution...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_scores = evaluation_data['all_similarities']
    top5_scores = evaluation_data['scores']
    
    ax.hist(all_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black', label='All Jobs')
    ax.axvline(np.mean(top5_scores), color='red', linestyle='--', linewidth=2.5, label=f'Top-5 Mean ({np.mean(top5_scores):.3f})')
    ax.axvline(np.mean(all_scores), color='orange', linestyle='--', linewidth=2.5, label=f'All Mean ({np.mean(all_scores):.3f})')
    
    ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Similarity Scores Across All Jobs\n' + f'(Total: {len(all_scores)} jobs)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    graph2_path = os.path.join(output_dir, '02_similarity_distribution.png')
    plt.savefig(graph2_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {graph2_path}")
    plt.show()
    
    # ========== Graph 3: Top Skills in Dataset ==========
    print("\n📌 Generating Graph 3: Top Skills in Dataset...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract all skills from dataset
    all_skills = []
    for skills_str in system.jobs_data['required_skills']:
        if isinstance(skills_str, str):
            skills = [s.strip().lower() for s in skills_str.split(',')]
            all_skills.extend(skills)
    
    # Count skills
    skill_counts = Counter(all_skills)
    top_skills = dict(skill_counts.most_common(15))
    
    skills = list(top_skills.keys())
    counts = list(top_skills.values())
    colors_skills = plt.cm.Set3(np.linspace(0, 1, len(skills)))
    
    bars = ax.barh(skills, counts, color=colors_skills, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(count + 10, bar.get_y() + bar.get_height()/2, 
                f'{int(count)}', va='center', fontweight='bold')
    
    ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Top 15 Most Common Skills in Dataset\n' + f'(Total {len(all_skills)} skill mentions)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    graph3_path = os.path.join(output_dir, '03_top_skills.png')
    plt.savefig(graph3_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {graph3_path}")
    plt.show()
    
    print(f"\n✓ All graphs generated successfully!")
    print(f"  • Location: {os.path.abspath(output_dir)}/")


# ============================================================================
# PHASE 8: RESULTS & ANALYSIS
# ============================================================================

def print_analysis(system, recommendations, user_input, evaluation_data, inference_time):
    """Print detailed analysis and insights."""
    print("\n" + "="*80)
    print("PHASE 8: RESULTS & ANALYSIS")
    print("="*80)
    
    # Display recommendations
    display_recommendations(recommendations, show_matched_skills=True)
    
    # Analysis insights
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    best_job = recommendations[0]
    best_score = best_job['similarity_score']
    
    print(f"\n🏆 Top Recommendation:")
    print(f"  • Job: {best_job['job_title']}")
    print(f"  • Company: {best_job['company']}")
    print(f"  • Match Score: {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"  • Why? Strong alignment between your skills and job requirements")
    
    # Skill analysis
    user_skills = set(s.strip().lower() for s in user_input.split(','))
    matched_skills_all = []
    for rec in recommendations:
        if rec.get('matched_skills'):
            matched_skills_all.extend(rec['matched_skills'])
    
    print(f"\n💡 Key Insights:")
    print(f"  • Your skills: {', '.join(user_skills)}")
    print(f"  • Matched in recommendations: {', '.join(set(matched_skills_all)) if matched_skills_all else 'N/A'}")
    print(f"  • Average match quality: {evaluation_data['avg_top5']:.1%}")
    print(f"  • System inference time: {inference_time*1000:.2f}ms")
    
    # Dataset insights
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total jobs analyzed: {len(system.jobs_data)}")
    print(f"  • Feature dimensions: {system.job_vectors.shape[1]}")
    print(f"  • Best match percentile: {(np.sum(evaluation_data['all_similarities'] <= evaluation_data['max_score']) / len(evaluation_data['all_similarities'])) * 100:.1f}%")
    
    print(f"\n✅ System Summary:")
    print(f"  • Status: {evaluation_data['quality']}")
    print(f"  • Recommendation quality: {evaluation_data['avg_top5']:.1%}")
    print(f"  • Processing time: {inference_time*1000:.2f}ms")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main orchestration function - controls complete pipeline."""
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║          KAGGLE JOB RECOMMENDATION SYSTEM - COMPLETE ML PIPELINE              ║")
    print("║                  TF-IDF + Cosine Similarity + Real Data                       ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    
    # Phase 1: Load Dataset
    df = load_dataset('data/job_recommendation_dataset.csv')
    
    # Phase 2: Preprocess
    df = preprocess_data(df)
    
    # Phase 3: Initialize system and vectorize
    system = JobRecommendationSystem()
    system.load_data('data/job_recommendation_dataset.csv')
    job_vectors = feature_engineering(system, df)
    
    # Phase 4: User Input
    user_input = get_user_input()
    
    # Phase 5: Generate Recommendations
    recommendations, inference_time = generate_recommendations(system, user_input)
    
    # Phase 6: Evaluate
    evaluation_data = evaluate_results(system, recommendations, user_input)
    
    # Phase 7: Generate Graphs
    generate_graphs(system, recommendations, evaluation_data, user_input)
    
    # Phase 8: Analysis
    print_analysis(system, recommendations, user_input, evaluation_data, inference_time)
    
    # Final message
    print("\n" + "="*80)
    print("🎉 PIPELINE EXECUTION COMPLETE!")
    print("="*80)
    print("\nAll outputs have been generated:")
    print("  ✓ Recommendations generated")
    print("  ✓ Evaluation metrics calculated")
    print("  ✓ Graphs saved to 'graphs/' folder")
    print("  ✓ Analysis printed above")
    print("\n")


if __name__ == '__main__':
    main()
