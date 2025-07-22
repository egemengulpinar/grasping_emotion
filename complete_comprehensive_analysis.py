#!/usr/bin/env python3
"""
Complete Comprehensive Research Analysis System
All outputs in English, systematic approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
from scipy import stats

def parse_analysis_report(file_path, participant_id):
    """Parse analysis report and extract all relevant data, including velocity."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    group = 'SEE' if int(participant_id[2:]) <= 10 else 'BLIND'
    objects = []
    
    # Split by object sections
    object_sections = re.split(r'\n\s*Id\d+/(\w+) Object:', content)
    
    for i in range(1, len(object_sections), 2):
        if i + 1 < len(object_sections):
            object_name = object_sections[i].strip()
            object_content = object_sections[i + 1]
            
            # Extract all metrics
            freq_match = re.search(r'Grasping frequency: ([\d.]+)/s', object_content)
            events_match = re.search(r'Grasping events: (\d+)', object_content)
            # Use a more robust regex for velocity that finds avg or max
            vel_match = re.search(r'(?:Max|Average|Mean) velocity: ([\d.]+) px/frame', object_content)
            emotion_match = re.search(r'Emotion felt: (\w+)', object_content)
            intensity_match = re.search(r'Intensity: (\w+)', object_content)
            comfort_match = re.search(r'Comfort during grasping: ([^,\n]+)', object_content)
            
            if freq_match:
                objects.append({
                    'participant_id': participant_id,
                    'group': group,
                    'object_type': object_name.lower(),
                    'grasping_frequency': float(freq_match.group(1)),
                    'grasping_events': int(events_match.group(1)) if events_match else None,
                    'velocity': float(vel_match.group(1)) if vel_match else 0.0, # Add velocity
                    'emotion': emotion_match.group(1) if emotion_match else None,
                    'intensity': intensity_match.group(1) if intensity_match else None,
                    'comfort': comfort_match.group(1).strip() if comfort_match else None
                })
    
    return objects

def load_all_data():
    """Load all participant data"""
    print("📁 Loading analysis data from all participants...")
    all_data = []
    
    for i in range(1, 21):
        file_path = f"analysis_results/ID{i}/analysis_report.txt"
        if os.path.exists(file_path):
            data = parse_analysis_report(file_path, f"ID{i}")
            all_data.extend(data)
            print(f"✓ ID{i}: {len(data)} objects")
        else:
            print(f"✗ Missing: ID{i}")
    
    df = pd.DataFrame(all_data)
    print(f"📊 Total: {len(df)} interactions from {df['participant_id'].nunique()} participants")
    return df

def compute_comprehensive_statistics(df):
    """Compute all statistics needed for analysis"""
    print("\n🔢 Computing comprehensive statistics...")
    
    # Separate groups
    see_data = df[df['group'] == 'SEE']
    blind_data = df[df['group'] == 'BLIND']
    
    # Participant-level means (correct for group comparison)
    see_participant_means = see_data.groupby('participant_id')['grasping_frequency'].mean()
    blind_participant_means = blind_data.groupby('participant_id')['grasping_frequency'].mean()
    
    # Statistical tests
    from scipy.stats import mannwhitneyu, ttest_ind
    
    u_stat, p_mann = mannwhitneyu(see_participant_means, blind_participant_means, alternative='two-sided')
    t_stat, p_ttest = ttest_ind(see_participant_means, blind_participant_means)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(see_participant_means)-1)*see_participant_means.var() + 
                         (len(blind_participant_means)-1)*blind_participant_means.var()) / 
                        (len(see_participant_means) + len(blind_participant_means) - 2))
    cohens_d = (see_participant_means.mean() - blind_participant_means.mean()) / pooled_std
    
    # Compile comprehensive statistics
    stats_dict = {
        'group_comparison': {
            'see_group': {
                'mean': see_participant_means.mean(),
                'std': see_participant_means.std(),
                'n': len(see_participant_means),
                'median': see_participant_means.median(),
                'min': see_participant_means.min(),
                'max': see_participant_means.max(),
                'individual_means': see_participant_means.to_dict()
            },
            'blind_group': {
                'mean': blind_participant_means.mean(),
                'std': blind_participant_means.std(),
                'n': len(blind_participant_means),
                'median': blind_participant_means.median(),
                'min': blind_participant_means.min(),
                'max': blind_participant_means.max(),
                'individual_means': blind_participant_means.to_dict()
            },
            'statistical_tests': {
                'mann_whitney_u': u_stat,
                'mann_whitney_p': p_mann,
                'ttest_statistic': t_stat,
                'ttest_p': p_ttest,
                'cohens_d': cohens_d,
                'effect_size': 'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
            }
        },
        'object_analysis': {},
        'emotion_analysis': {}
    }
    
    # Add velocity stats
    stats_dict['velocity_analysis'] = {
        'overall_mean': df['velocity'].mean(),
        'overall_std': df['velocity'].std()
    }

    # Object-specific analysis
    for obj in df['object_type'].unique():
        obj_all = df[df['object_type'] == obj]
        obj_see = see_data[see_data['object_type'] == obj]
        obj_blind = blind_data[blind_data['object_type'] == obj]
        
        obj_stats = {
            'overall': {
                'mean': obj_all['grasping_frequency'].mean(),
                'std': obj_all['grasping_frequency'].std(),
                'count': len(obj_all)
            },
            'see_group': {
                'mean': obj_see['grasping_frequency'].mean() if len(obj_see) > 0 else None,
                'std': obj_see['grasping_frequency'].std() if len(obj_see) > 0 else None,
                'count': len(obj_see)
            },
            'blind_group': {
                'mean': obj_blind['grasping_frequency'].mean() if len(obj_blind) > 0 else None,
                'std': obj_blind['grasping_frequency'].std() if len(obj_blind) > 0 else None,
                'count': len(obj_blind)
            }
        }
        
        # Object-specific statistical test
        if len(obj_see) > 0 and len(obj_blind) > 0:
            obj_u, obj_p = mannwhitneyu(obj_see['grasping_frequency'], obj_blind['grasping_frequency'], alternative='two-sided')
            obj_stats['statistical_test'] = {'u_statistic': obj_u, 'p_value': obj_p}
        
        stats_dict['object_analysis'][obj] = obj_stats
    
    # Emotion analysis
    if 'emotion' in df.columns:
        emotion_data = df.dropna(subset=['emotion'])
        if len(emotion_data) > 0:
            emotion_stats = emotion_data.groupby('emotion')['grasping_frequency'].agg(['mean', 'std', 'count']).to_dict('index')
            velocity_emotion_stats = emotion_data.groupby('emotion')['velocity'].agg(['mean', 'std', 'count']).to_dict('index')
            stats_dict['emotion_analysis'] = {'frequency': emotion_stats, 'velocity': velocity_emotion_stats}
    
    # Print key results
    see_stats = stats_dict['group_comparison']['see_group']
    blind_stats = stats_dict['group_comparison']['blind_group']
    test_stats = stats_dict['group_comparison']['statistical_tests']
    
    print(f"   SEE Group: {see_stats['mean']:.2f} ± {see_stats['std']:.2f} events/s (n={see_stats['n']})")
    print(f"   BLIND Group: {blind_stats['mean']:.2f} ± {blind_stats['std']:.2f} events/s (n={blind_stats['n']})")
    print(f"   Difference: {see_stats['mean'] - blind_stats['mean']:.2f} events/s")
    print(f"   Statistical Test: p = {test_stats['mann_whitney_p']:.4f}")
    print(f"   Effect Size: Cohen's d = {test_stats['cohens_d']:.3f} ({test_stats['effect_size']})")
    
    return stats_dict

def create_main_analysis_plot(df, output_dir):
    """Create 1_main_analysis.png - Object engagement hierarchy"""
    print("\n📊 Creating main analysis plot (1_main_analysis.png)...")
    
    plt.figure(figsize=(12, 8))
    
    # Calculate object statistics
    object_stats = df.groupby('object_type')['grasping_frequency'].agg(['mean', 'std', 'count'])
    object_stats = object_stats.sort_values('mean', ascending=True)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(object_stats))
    bars = plt.barh(y_pos, object_stats['mean'], xerr=object_stats['std'], 
                   color='steelblue', alpha=0.8, capsize=6, edgecolor='navy', linewidth=1)
    
    # Customize plot
    plt.yticks(y_pos, [obj.title() for obj in object_stats.index], fontsize=12)
    plt.xlabel('Grasping Frequency (events/second)', fontsize=14, fontweight='bold')
    plt.title('Hand Movement Analysis: Grasping Frequency by Object Type\n(Mean ± Standard Deviation)', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (mean_val, std_val, count_val) in enumerate(zip(object_stats['mean'], object_stats['std'], object_stats['count'])):
        plt.text(mean_val + std_val + 0.3, i, f'{mean_val:.1f}\n(n={count_val})', 
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.xlim(0, max(object_stats['mean'] + object_stats['std']) * 1.15)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/1_main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Main analysis plot saved")

def create_grasping_analysis_plot(df, stats_dict, output_dir):
    """Create 2_grasping_analysis.png - Comprehensive 4-panel analysis"""
    print("\n📊 Creating grasping analysis plot (2_grasping_analysis.png)...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: Group Comparison Boxplot
    see_means = pd.Series(stats_dict['group_comparison']['see_group']['individual_means'])
    blind_means = pd.Series(stats_dict['group_comparison']['blind_group']['individual_means'])
    
    bp1 = ax1.boxplot([see_means.values, blind_means.values], 
                     labels=['SEE Group\n(n=10)', 'BLIND Group\n(n=10)'], 
                     patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightcoral')
    bp1['boxes'][0].set_alpha(0.8)
    bp1['boxes'][1].set_alpha(0.8)
    
    ax1.set_ylabel('Grasping Frequency (events/s)', fontsize=12)
    ax1.set_title('A. Group Comparison: SEE vs BLIND', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistical info
    p_val = stats_dict['group_comparison']['statistical_tests']['mann_whitney_p']
    effect_size = stats_dict['group_comparison']['statistical_tests']['effect_size']
    ax1.text(0.5, 0.95, f'p = {p_val:.3f} ({effect_size} effect)', 
            transform=ax1.transAxes, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Panel 2: Object-Group Interaction
    obj_group_data = df.groupby(['object_type', 'group'])['grasping_frequency'].mean().unstack()
    obj_group_data.plot(kind='bar', ax=ax2, color=['lightblue', 'lightcoral'], alpha=0.8, width=0.8)
    ax2.set_ylabel('Grasping Frequency (events/s)', fontsize=12)
    ax2.set_title('B. Object-Specific Group Differences', fontweight='bold', fontsize=14)
    ax2.legend(title='Group', loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Individual Participant Performance
    participants = sorted(df['participant_id'].unique(), key=lambda x: int(x[2:]))
    participant_means = df.groupby('participant_id')['grasping_frequency'].mean()
    
    colors = ['lightblue' if int(p[2:]) <= 10 else 'lightcoral' for p in participants]
    bars = ax3.bar(range(len(participants)), [participant_means[p] for p in participants], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xticks(range(len(participants)))
    ax3.set_xticklabels([p.replace('ID', '') for p in participants], fontsize=10)
    ax3.set_xlabel('Participant ID', fontsize=12)
    ax3.set_ylabel('Grasping Frequency (events/s)', fontsize=12)
    ax3.set_title('C. Individual Participant Performance', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add group separator
    ax3.axvline(x=9.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.text(4.5, max([participant_means[p] for p in participants]) * 0.95, 'SEE', 
            ha='center', fontweight='bold', fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.text(14.5, max([participant_means[p] for p in participants]) * 0.95, 'BLIND', 
            ha='center', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Panel 4: Emotion-Performance Relationship
    emotion_data = df.dropna(subset=['emotion'])
    if len(emotion_data) > 0:
        emotion_stats = emotion_data.groupby('emotion')['grasping_frequency'].agg(['mean', 'std', 'count'])
        emotion_stats = emotion_stats.sort_values('mean', ascending=False)
        
        bars = ax4.bar(range(len(emotion_stats)), emotion_stats['mean'], 
                      yerr=emotion_stats['std'], capsize=5, alpha=0.8, 
                      color='lightgreen', edgecolor='darkgreen')
        
        ax4.set_xticks(range(len(emotion_stats)))
        ax4.set_xticklabels(emotion_stats.index, rotation=45, fontsize=10)
        ax4.set_ylabel('Grasping Frequency (events/s)', fontsize=12)
        ax4.set_title('D. Emotion vs Grasping Performance', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (mean_val, count_val) in enumerate(zip(emotion_stats['mean'], emotion_stats['count'])):
            ax4.text(i, mean_val + emotion_stats['std'].iloc[i] + 0.3, f'n={count_val}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('3-Category Movement System Analysis: Comprehensive Results\n(Grasping, Holding, Other)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/2_grasping_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Grasping analysis plot saved")

def create_summary_table_latex(df, stats_dict, output_dir):
    """Create 4_summary_table.tex - LaTeX formatted summary table"""
    print("\n📋 Creating summary table (4_summary_table.tex)...")
    
    see_stats = stats_dict['group_comparison']['see_group']
    blind_stats = stats_dict['group_comparison']['blind_group']
    test_stats = stats_dict['group_comparison']['statistical_tests']
    
    latex_content = f"""
% Hand Movement Analysis Summary Tables
% Generated by Comprehensive Research Analysis System

\\begin{{table}}[htbp]
\\centering
\\caption{{Comprehensive Hand Movement Analysis: Group Comparison Summary}}
\\label{{tab:group_comparison}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Measure}} & \\textbf{{SEE Group}} & \\textbf{{BLIND Group}} & \\textbf{{Difference}} & \\textbf{{Statistical Test}} \\\\
\\hline
Sample Size & {see_stats['n']} & {blind_stats['n']} & - & - \\\\
Mean ± SD & {see_stats['mean']:.2f} ± {see_stats['std']:.2f} & {blind_stats['mean']:.2f} ± {blind_stats['std']:.2f} & {see_stats['mean'] - blind_stats['mean']:.2f} & p = {test_stats['mann_whitney_p']:.3f} \\\\
Median & {see_stats['median']:.2f} & {blind_stats['median']:.2f} & {see_stats['median'] - blind_stats['median']:.2f} & - \\\\
Range & {see_stats['min']:.2f} - {see_stats['max']:.2f} & {blind_stats['min']:.2f} - {blind_stats['max']:.2f} & - & - \\\\
Effect Size & \\multicolumn{{4}}{{c}}{{Cohen's d = {test_stats['cohens_d']:.3f} ({test_stats['effect_size']} effect)}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Object-Specific Analysis: SEE vs BLIND Group Comparisons}}
\\label{{tab:object_analysis}}
\\begin{{tabular}}{{lcccccc}}
\\hline
\\textbf{{Object}} & \\multicolumn{{2}}{{c}}{{\\textbf{{SEE Group}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{BLIND Group}}}} & \\textbf{{Difference}} & \\textbf{{p-value}} \\\\
 & Mean & SD & Mean & SD & & \\\\
\\hline
"""
    
    # Add object-specific data
    for obj_name in sorted(stats_dict['object_analysis'].keys()):
        obj_stats = stats_dict['object_analysis'][obj_name]
        see_mean = obj_stats['see_group']['mean']
        see_std = obj_stats['see_group']['std']
        blind_mean = obj_stats['blind_group']['mean']
        blind_std = obj_stats['blind_group']['std']
        
        if see_mean is not None and blind_mean is not None:
            difference = see_mean - blind_mean
            p_val = obj_stats.get('statistical_test', {}).get('p_value', 'N/A')
            p_val_str = f"{p_val:.3f}" if isinstance(p_val, float) else p_val
            
            latex_content += f"{obj_name.title()} & {see_mean:.2f} & {see_std:.2f} & {blind_mean:.2f} & {blind_std:.2f} & {difference:.2f} & {p_val_str} \\\\\n"
    
    latex_content += """\\hline
\\end{tabular}
\\end{table}

\\begin{table}[htbp]
\\centering
\\caption{Object Engagement Hierarchy}
\\label{tab:object_hierarchy}
\\begin{tabular}{lccc}
\\hline
\\textbf{Object Type} & \\textbf{Mean Frequency} & \\textbf{Std Deviation} & \\textbf{Rank} \\\\
\\hline
"""
    
    # Add object hierarchy
    object_hierarchy = [(obj, stats['overall']['mean'], stats['overall']['std']) 
                       for obj, stats in stats_dict['object_analysis'].items()]
    object_hierarchy.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (obj_name, mean_freq, std_freq) in enumerate(object_hierarchy, 1):
        latex_content += f"{obj_name.title()} & {mean_freq:.2f} & {std_freq:.2f} & {rank} \\\\\n"
    
    latex_content += """\\hline
\\multicolumn{4}{l}{\\textit{Note: Frequency measured in events per second}} \\\\
\\multicolumn{4}{l}{\\textit{SEE Group: Participants ID01-ID10, BLIND Group: ID11-ID20}} \\\\
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX file
    with open(f'{output_dir}/4_summary_table.tex', 'w') as f:
        f.write(latex_content)
    
    print("✓ Summary table saved as LaTeX")

def main():
    """Main comprehensive analysis pipeline"""
    print("🔬 COMPLETE COMPREHENSIVE RESEARCH ANALYSIS SYSTEM")
    print("="*60)
    print("Systematic analysis for scientific publication")
    print("All outputs generated in English language")
    print()
    
    # Create output directory
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load all data
    df = load_all_data()
    
    if len(df) == 0:
        print("❌ No data loaded. Check analysis_results directory.")
        return
    
    # Step 2: Compute comprehensive statistics
    stats_dict = compute_comprehensive_statistics(df)
    
    # Step 3: Create plots and tables
    create_main_analysis_plot(df, output_dir)
    create_grasping_analysis_plot(df, stats_dict, output_dir)
    create_summary_table_latex(df, stats_dict, output_dir)
    
    # Step 4: Save all raw data and statistics to a single JSON file for LLM analysis
    llm_output = {
        "study_metadata": {
            "total_participants": df['participant_id'].nunique(),
            "total_interactions": len(df),
            "groups": [
                f"SEE (ID1-ID10)",
                f"BLIND (ID11-ID20)"
            ]
        },
        "statistics": stats_dict,
        # Ensure the full dataframe with the 'velocity' column is saved
        "participant_data": df.to_dict(orient='records')
    }
    
    with open(f'{output_dir}/llm_analysis_input.json', 'w') as f:
        # Use default=str to handle potential numpy types that are not JSON serializable
        json.dump(llm_output, f, indent=2, default=str)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("============================================================")
    print("📁 All files saved to: analysis_results/")
    print("\n📋 Generated Files:")
    print("  • 1_main_analysis.png - Object engagement hierarchy")
    print("  • 2_grasping_analysis.png - 4-panel comprehensive analysis")
    print("  • 4_summary_table.tex - LaTeX summary tables")
    print("  • llm_analysis_input.json - Data for LLM analysis")
    print("\n🎯 Ready for scientific publication and advanced LLM analysis!")
    
    # Print key results summary
    see_stats = stats_dict['group_comparison']['see_group']
    blind_stats = stats_dict['group_comparison']['blind_group']
    print(f"\n📊 KEY RESULTS SUMMARY:")
    print(f"   SEE Group: {see_stats['mean']:.2f} ± {see_stats['std']:.2f} events/s")
    print(f"   BLIND Group: {blind_stats['mean']:.2f} ± {blind_stats['std']:.2f} events/s")
    print(f"   Statistical significance: p = {stats_dict['group_comparison']['statistical_tests']['mann_whitney_p']:.4f}")

if __name__ == "__main__":
    main()
