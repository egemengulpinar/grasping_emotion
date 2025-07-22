#!/usr/bin/env python3
"""
🤖 LLM Analysis Demo Script
===========================

This script demonstrates how the LLM-optimized analysis files can be easily
processed for scientific research and comprehensive insights.

Usage:
python3 llm_analysis_demo.py analysis_results/llm_analysis/
"""

import pandas as pd
import json
import os
import sys

def analyze_llm_optimized_data(llm_analysis_dir):
    """
    Demonstrates how to analyze the LLM-optimized data files
    """
    print("🤖 LLM-Optimized Hand Movement Analysis")
    print("=" * 50)
    
    # Load all the LLM-optimized files
    files = {
        'master_summary': os.path.join(llm_analysis_dir, 'master_video_summary.csv'),
        'key_findings': os.path.join(llm_analysis_dir, 'key_findings_summary.json'),
        'statistical_analysis': os.path.join(llm_analysis_dir, 'statistical_analysis.json'),
        'plot_data': os.path.join(llm_analysis_dir, 'plot_data_for_llm.json'),
        'markdown_report': os.path.join(llm_analysis_dir, 'llm_analysis_report.md')
    }
    
    # Check if files exist
    for file_type, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return
    
    print("✅ All LLM-optimized files found!")
    print()
    
    # 1. Load and analyze master summary
    print("📊 MASTER DATASET ANALYSIS")
    print("-" * 30)
    
    master_df = pd.read_csv(files['master_summary'])
    print(f"Videos analyzed: {len(master_df)}")
    print(f"Total duration: {master_df['duration_seconds'].sum():.1f} seconds")
    print(f"Total grasping events: {master_df['grasping_events_count'].sum():,}")
    print()
    
    # Top performing objects
    print("🏆 TOP PERFORMING OBJECTS (by grasping frequency):")
    top_objects = master_df.nlargest(3, 'grasping_frequency')[['video_id', 'grasping_frequency', 'most_common_gesture']]
    for idx, row in top_objects.iterrows():
        print(f"  {row['video_id'].title()}: {row['grasping_frequency']:.1f} events/s ({row['most_common_gesture']})")
    print()
    
    # 2. Load key findings
    print("🔍 KEY FINDINGS SUMMARY")
    print("-" * 30)
    
    with open(files['key_findings'], 'r') as f:
        key_findings = json.load(f)
    
    overview = key_findings['study_overview']
    rankings = key_findings['performance_rankings']
    
    print(f"Study scope: {overview['analysis_scope']}")
    print(f"Most engaging object: {rankings['most_engaging'].title()}")
    print(f"Least engaging object: {rankings['least_engaging'].title()}")
    print(f"Engagement ratio: {rankings['engagement_ratio']}x difference")
    print()
    
    # Research implications
    print("🧠 RESEARCH IMPLICATIONS:")
    for implication in key_findings['research_implications']:
        print(f"  • {implication}")
    print()
    
    # 3. Statistical analysis insights
    print("📈 STATISTICAL INSIGHTS")
    print("-" * 30)
    
    with open(files['statistical_analysis'], 'r') as f:
        stats = json.load(f)
    
    desc_stats = stats['descriptive_stats']
    correlations = stats['correlations']
    
    print("Descriptive Statistics:")
    print(f"  • Grasping Frequency: {desc_stats['grasping_frequencies']['mean']:.2f} ± {desc_stats['grasping_frequencies']['std']:.2f} events/s")
    print(f"  • Hand Velocity: {desc_stats['avg_velocities']['mean']:.3f} ± {desc_stats['avg_velocities']['std']:.3f} px/s")
    print(f"  • Hand Openness: {desc_stats['hand_openness']['mean']:.3f} ± {desc_stats['hand_openness']['std']:.3f}")
    print()
    
    print("Significant Correlations:")
    for corr_name, corr_data in correlations.items():
        if corr_data['significance'] == 'significant':
            metric_names = corr_name.replace('_vs_', ' vs ').replace('_', ' ').title()
            print(f"  • {metric_names}: r = {corr_data['correlation']:.3f}, p = {corr_data['p_value']:.4f}")
    print()
    
    # 4. Object difficulty ranking
    print("🎯 OBJECT DIFFICULTY RANKING")
    print("-" * 30)
    
    rankings = stats['object_rankings']['engagement_ranking']
    print("| Rank | Object | Score | Level |")
    print("|------|--------|-------|-------|")
    
    for ranking in rankings:
        level = "High" if ranking['score'] > 0.7 else "Medium" if ranking['score'] > 0.4 else "Low"
        print(f"| {ranking['rank']} | {ranking['object'].title()} | {ranking['score']:.3f} | {level} |")
    print()
    
    # 5. Temporal patterns insight
    print("⏱️ TEMPORAL PATTERNS")
    print("-" * 30)
    
    with open(files['plot_data'], 'r') as f:
        plot_data = json.load(f)
    
    temporal_patterns = plot_data['temporal_patterns']
    
    print("Peak Activity Analysis:")
    for video, pattern in temporal_patterns.items():
        peak_bin = pattern['peak_activity_bin']
        trend = pattern['activity_trend']
        print(f"  • {video.title()}: Peak in time bin {peak_bin}/5, trend: {trend}")
    print()
    
    # 6. Movement characteristics
    print("🤲 MOVEMENT CHARACTERISTICS")
    print("-" * 30)
    
    trajectory_data = plot_data['trajectory_analysis']
    
    print("Movement Area Analysis (normalized):")
    for video, traj in trajectory_data.items():
        area = traj['movement_area']
        length = traj['trajectory_length']
        print(f"  • {video.title()}: Area = {area:.4f}, Frames = {length:,}")
    print()
    
    # 7. Clinical recommendations
    print("🏥 CLINICAL RECOMMENDATIONS")
    print("-" * 30)
    
    # Generate recommendations based on data
    high_engagement = [obj for obj in key_findings['object_characteristics'] if obj['engagement_level'] == 'high']
    medium_engagement = [obj for obj in key_findings['object_characteristics'] if obj['engagement_level'] == 'medium']
    low_engagement = [obj for obj in key_findings['object_characteristics'] if obj['engagement_level'] == 'low']
    
    print("Based on engagement analysis:")
    print(f"  • HIGH engagement objects ({len(high_engagement)}): Best for motor assessment")
    for obj in high_engagement:
        print(f"    - {obj['object'].title()}: {obj['grasping_frequency']:.1f} events/s")
    
    print(f"  • MEDIUM engagement objects ({len(medium_engagement)}): Good for training")
    for obj in medium_engagement:
        print(f"    - {obj['object'].title()}: {obj['grasping_frequency']:.1f} events/s")
    
    print(f"  • LOW engagement objects ({len(low_engagement)}): May need modification")
    for obj in low_engagement:
        print(f"    - {obj['object'].title()}: {obj['grasping_frequency']:.1f} events/s")
    print()
    
    # 8. Data summary for LLM
    print("📋 LLM DATA SUMMARY")
    print("-" * 30)
    print("Files optimized for LLM analysis:")
    print(f"  ✅ master_video_summary.csv - {len(master_df)} videos × {len(master_df.columns)} metrics")
    print(f"  ✅ key_findings_summary.json - Structured insights")
    print(f"  ✅ statistical_analysis.json - Statistical test results")
    print(f"  ✅ plot_data_for_llm.json - Text-based visualization data")
    print(f"  ✅ llm_analysis_report.md - Comprehensive markdown report")
    print()
    print("🎯 Ready for LLM analysis and scientific publication!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 llm_analysis_demo.py <llm_analysis_directory>")
        print("Example: python3 llm_analysis_demo.py analysis_results/llm_analysis/")
        return
    
    llm_analysis_dir = sys.argv[1]
    
    if not os.path.exists(llm_analysis_dir):
        print(f"❌ Directory not found: {llm_analysis_dir}")
        return
    
    analyze_llm_optimized_data(llm_analysis_dir)

if __name__ == "__main__":
    main() 