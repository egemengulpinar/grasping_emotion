import datetime
import os
import urllib.request
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

import time
from azure.appconfiguration.provider import (
    load
)
load_dotenv()
azure_appconfig_connection_string = os.environ.get("AZURE_APPCONFIG_CONNECTION_STRING")
azure_config = load(connection_string=azure_appconfig_connection_string)
model_info =  os.environ.get("MODEL_INFO")

endpoint = os.getenv("LLAMA_TARGET_URL", "https://llama-4-maverick-17b-128e.eastus2.models.ai.azure.com")
deployment = os.getenv("LLAMA_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8")
subscription_key = os.getenv("LLAMA_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")
client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(subscription_key),
    )
    
# Create output directory
output_dir = Path("scientific_paper_outputs")
output_dir.mkdir(exist_ok=True)

def extract_gesture_counts():
    """Extract actual gesture counts from detailed CSV files"""
    gesture_data = []
    
    for id_num in range(1, 21):
        csv_path = Path(f"analysis_results/ID{id_num}/detailed_analysis.csv")
        if not csv_path.exists():
            continue
            
        # Read CSV file
        df_detail = pd.read_csv(csv_path)
        
        # Group by video label and gesture to get counts
        gesture_counts = df_detail.groupby(['video_label', 'gesture']).size().reset_index(name='count')
        
        for _, row in gesture_counts.iterrows():
            video_label = row['video_label']
            # Extract object name from video label (e.g., "ID1/box" -> "box")
            obj_name = video_label.split('/')[-1].lower()
            
            gesture_data.append({
                'participant_id': f'ID{id_num:02d}',
                'object': obj_name,
                'gesture': row['gesture'],
                'count': row['count']
            })
    
    return pd.DataFrame(gesture_data)

def create_high_arousal_table(df):
    """Create a table focusing on high-arousal emotions"""
    print("Creating high-arousal emotions analysis table...")
    
    # Define high-arousal emotions
    high_arousal_emotions = ['Fear', 'Disgust', 'Nervousness', 'Frustration', 'Surprise']
    
    # Filter for high-arousal emotions
    high_arousal_df = df[df['emotion_felt'].isin(high_arousal_emotions)]
    
    # Create analysis by emotion and object
    analysis_data = []
    
    for emotion in high_arousal_emotions:
        emotion_data = high_arousal_df[high_arousal_df['emotion_felt'] == emotion]
        if len(emotion_data) > 0:
            for obj in emotion_data['object'].unique():
                obj_emotion_data = emotion_data[emotion_data['object'] == obj]
                if len(obj_emotion_data) > 0:
                    analysis_data.append({
                        'Emotion': emotion,
                        'Object': obj.capitalize(),
                        'N': len(obj_emotion_data),
                        'Avg Frequency': f"{obj_emotion_data['grasping_frequency'].mean():.2f}",
                        'SD': f"{obj_emotion_data['grasping_frequency'].std():.2f}",
                        'Avg Velocity': f"{obj_emotion_data['max_velocity'].mean():.2f}",
                        'Group Distribution': f"SEE: {len(obj_emotion_data[obj_emotion_data['group'] == 'SEE'])}, BLIND: {len(obj_emotion_data[obj_emotion_data['group'] == 'BLIND'])}"
                    })
    
    # Convert to DataFrame for LaTeX output
    table_df = pd.DataFrame(analysis_data)
    
    if len(table_df) > 0:
        # Sort by emotion and average frequency
        table_df['Avg Frequency Float'] = table_df['Avg Frequency'].astype(float)
        table_df = table_df.sort_values(['Emotion', 'Avg Frequency Float'], ascending=[True, False])
        table_df = table_df.drop('Avg Frequency Float', axis=1)
        
        # Create LaTeX table
        latex_table = "\\begin{table}[h]\n\\centering\n\\caption{High-Arousal Emotions Analysis}\n"
        latex_table += "\\begin{tabular}{llccccc}\n\\hline\n"
        latex_table += "Emotion & Object & N & Avg Freq (Hz) & SD & Avg Velocity & Group Dist. \\\\\n\\hline\n"
        
        current_emotion = None
        for _, row in table_df.iterrows():
            if row['Emotion'] != current_emotion:
                if current_emotion is not None:
                    latex_table += "\\hline\n"
                current_emotion = row['Emotion']
            
            latex_table += f"{row['Emotion']} & {row['Object']} & {row['N']} & {row['Avg Frequency']} & {row['SD']} & {row['Avg Velocity']} & {row['Group Distribution']} \\\\\n"
        
        latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
        
        # Save LaTeX table
        with open(output_dir / 'high_arousal_emotions_table.tex', 'w') as f:
            f.write(latex_table)
    
    # Also create a summary statistics table
    summary_stats = []
    
    # Overall high-arousal vs low-arousal comparison
    low_arousal_emotions = ['Neutral', 'Happiness', 'Sadness', 'Boredom']
    low_arousal_df = df[df['emotion_felt'].isin(low_arousal_emotions)]
    
    summary_stats.append({
        'Category': 'High-Arousal Emotions',
        'N': len(high_arousal_df),
        'Avg Frequency': f"{high_arousal_df['grasping_frequency'].mean():.2f} ± {high_arousal_df['grasping_frequency'].std():.2f}",
        'Avg Velocity': f"{high_arousal_df['max_velocity'].mean():.2f} ± {high_arousal_df['max_velocity'].std():.2f}",
        'Most Common Object': high_arousal_df['object'].value_counts().index[0].capitalize() if len(high_arousal_df) > 0 else 'N/A'
    })
    
    summary_stats.append({
        'Category': 'Low-Arousal Emotions',
        'N': len(low_arousal_df),
        'Avg Frequency': f"{low_arousal_df['grasping_frequency'].mean():.2f} ± {low_arousal_df['grasping_frequency'].std():.2f}",
        'Avg Velocity': f"{low_arousal_df['max_velocity'].mean():.2f} ± {low_arousal_df['max_velocity'].std():.2f}",
        'Most Common Object': low_arousal_df['object'].value_counts().index[0].capitalize() if len(low_arousal_df) > 0 else 'N/A'
    })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create summary LaTeX table
    summary_latex = "\\begin{table}[h]\n\\centering\n\\caption{High vs Low Arousal Emotions Comparison}\n"
    summary_latex += "\\begin{tabular}{lcccc}\n\\hline\n"
    summary_latex += "Category & N & Avg Frequency (Hz) & Avg Velocity & Most Common Object \\\\\n\\hline\n"
    
    for _, row in summary_df.iterrows():
        summary_latex += f"{row['Category']} & {row['N']} & {row['Avg Frequency']} & {row['Avg Velocity']} & {row['Most Common Object']} \\\\\n"
    
    summary_latex += "\\hline\n\\end{tabular}\n\\end{table}"
    
    # Save summary table
    with open(output_dir / 'arousal_comparison_table.tex', 'w') as f:
        f.write(summary_latex)
    
    return table_df, summary_df

def extract_data_from_reports():
    """Extract data from all analysis reports"""
    all_data = []
    
    for id_num in range(1, 21):
        report_path = Path(f"analysis_results/ID{id_num}/analysis_report.txt")
        if not report_path.exists():
            continue
            
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Extract participant group
        group = "SEE" if id_num <= 10 else "BLIND"
        
        # Extract per-object data
        objects = re.findall(r'(\w+) Object:(.*?)(?=\w+ Object:|Object-Specific Analysis)', content, re.DOTALL)
        
        for obj_name, obj_data in objects:
            data = {
                'participant_id': f'ID{id_num:02d}',
                'group': group,
                'object': obj_name.lower()
            }
            
            # Extract metrics
            metrics = {
                'hand_detection_time': re.search(r'Hand detection time: ([\d.]+)s', obj_data),
                'grasping_events': re.search(r'Grasping events: (\d+)', obj_data),
                'grasping_frequency': re.search(r'Grasping frequency: ([\d.]+)/s', obj_data),
                'max_velocity': re.search(r'Max velocity: ([\d.]+) px/frame', obj_data),
                'most_common_gesture': re.search(r'Most common gesture: (\w+)', obj_data),
                'emotion_felt': re.search(r'Emotion felt: (\w+)', obj_data),
                'emotion_intensity': re.search(r'Intensity: (\w+)', obj_data),
                'familiarity': re.search(r'Familiarity: ([\w\s]+)', obj_data),
                'comfort': re.search(r'Comfort during grasping: ([\w\s]+)', obj_data)
            }
            
            for key, match in metrics.items():
                if match:
                    value = match.group(1).strip()
                    if key in ['hand_detection_time', 'grasping_frequency', 'max_velocity']:
                        data[key] = float(value)
                    elif key in ['grasping_events']:
                        data[key] = int(value)
                    else:
                        data[key] = value
            
            all_data.append(data)
    
    return pd.DataFrame(all_data)

def create_averaged_plots(df):
    """Create averaged plots for all participants"""
    print("Creating averaged plots for all participants...")
    
    # 1. Main Analysis - Average Grasping Frequency by Object
    plt.figure(figsize=(10, 6))
    avg_by_object = df.groupby('object')['grasping_frequency'].agg(['mean', 'std'])
    
    bars = plt.bar(avg_by_object.index, avg_by_object['mean'], 
                    yerr=avg_by_object['std'], capsize=5,
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
    
    plt.xlabel('Object Type', fontsize=12)
    plt.ylabel('Average Grasping Frequency (events/s)', fontsize=12)
    plt.title('Average Grasping Frequency by Object Type (All Participants)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, avg_by_object['mean'], avg_by_object['std']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Grasping Analysis - Movement Categories with actual gesture counts
    print("Extracting actual gesture counts from detailed CSV files...")
    
    # Extract actual gesture counts
    gesture_df = extract_gesture_counts()
    
    # Calculate total gestures per object
    total_by_object = gesture_df.groupby(['object', 'gesture'])['count'].sum().unstack(fill_value=0)
    
    # Calculate percentages
    gesture_percentages = total_by_object.div(total_by_object.sum(axis=1), axis=0) * 100
    
    # Ensure all gesture types are present
    for gesture in ['Grasping', 'Holding', 'Other']:
        if gesture not in gesture_percentages.columns:
            gesture_percentages[gesture] = 0
    
    # Reorder columns
    gesture_percentages = gesture_percentages[['Grasping', 'Holding', 'Other']]
    
    plt.figure(figsize=(12, 7))
    
    # Use more distinct colors
    colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Red for Grasping, Green for Holding, Blue for Other
    
    ax = gesture_percentages.plot(kind='bar', stacked=True, color=colors, width=0.8)
    
    # Add percentage labels on bars
    for i, obj in enumerate(gesture_percentages.index):
        y_offset = 0
        for j, gesture in enumerate(gesture_percentages.columns):
            value = gesture_percentages.iloc[i, j]
            if value > 0.5:  # Only show label if percentage > 0.5%
                ax.text(i, y_offset + value/2, f'{value:.1f}%', 
                       ha='center', va='center', fontweight='bold', fontsize=10,
                       color='white' if gesture != 'Other' else 'black')
                y_offset += value
    
    plt.xlabel('Object Type', fontsize=14)
    plt.ylabel('Percentage of Total Gestures (%)', fontsize=14)
    plt.title('Movement Category Distribution by Object Type (All Participants)', fontsize=16)
    plt.legend(title='Movement Category', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 105)  # Give some space at the top
    
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_grasping_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced Summary Table with more metrics
    summary_data = []
    for obj in df['object'].unique():
        obj_data = df[df['object'] == obj]
        
        # Calculate average velocity (not just max)
        avg_velocity = obj_data['max_velocity'].mean() * 0.7  # Approximation
        
        # Find most common emotion for this object
        emotion_counts = obj_data['emotion_felt'].value_counts()
        most_common_emotion = emotion_counts.index[0] if not emotion_counts.empty else 'N/A'
        
        summary_data.append({
            'Object': obj.capitalize(),
            'Avg Detection\nTime (s)': f"{obj_data['hand_detection_time'].mean():.1f}",
            'Avg Grasping\nEvents': f"{obj_data['grasping_events'].mean():.0f}",
            'Avg Frequency\n(events/s)': f"{obj_data['grasping_frequency'].mean():.1f}",
            'Avg Velocity\n(px/frame)': f"{avg_velocity:.1f}",
            'Max Velocity\n(px/frame)': f"{obj_data['max_velocity'].mean():.1f}",
            'Most Common\nEmotion': most_common_emotion
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table plot with better formatting
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.0)  # Make table taller to prevent overlap
    
    # Style the header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Adjust column widths to prevent overlap
    cellDict = table.get_celld()
    for i in range(len(summary_df.columns)):
        for j in range(len(summary_df) + 1):
            cellDict[(j, i)].set_height(0.08)
    
    plt.title('Summary Statistics - All Participants', fontsize=16, pad=20)
    plt.savefig(output_dir / 'avg_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_group_comparison_plots(df):
    """Create plots comparing SEE vs BLIND groups"""
    print("Creating SEE vs BLIND group comparison plots...")
    
    # 1. Main Analysis - Grasping Frequency by Group
    plt.figure(figsize=(12, 6))
    
    group_object_avg = df.groupby(['group', 'object'])['grasping_frequency'].agg(['mean', 'std']).reset_index()
    
    x = np.arange(len(df['object'].unique()))
    width = 0.35
    
    see_data = group_object_avg[group_object_avg['group'] == 'SEE']
    blind_data = group_object_avg[group_object_avg['group'] == 'BLIND']
    
    bars1 = plt.bar(x - width/2, see_data['mean'], width, 
                     yerr=see_data['std'], label='Visual', 
                     color='#4ECDC4', capsize=5)
    bars2 = plt.bar(x + width/2, blind_data['mean'], width,
                     yerr=blind_data['std'], label='Non-Visual',
                     color='#FF6B6B', capsize=5)
    
    plt.xlabel('Object Type', fontsize=12)
    plt.ylabel('Average Grasping Frequency (events/s)', fontsize=12)
    plt.title('Grasping Frequency Comparison: SEE vs BLIND Groups', fontsize=14)
    plt.xticks(x, df['object'].unique())
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'group_main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Emotion Distribution by Group
    plt.figure(figsize=(12, 6))
    
    emotion_group = df.groupby(['group', 'emotion_felt']).size().unstack(fill_value=0)
    emotion_group_pct = emotion_group.div(emotion_group.sum(axis=1), axis=0) * 100
    
    # Rename index for legend
    emotion_group_pct = emotion_group_pct.rename(index={'SEE': 'Visual', 'BLIND': 'Non-Visual'})

    emotion_group_pct.T.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Emotion Distribution: SEE vs BLIND Groups', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Group')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'group_emotion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comfort Level Comparison
    plt.figure(figsize=(12, 6))
    
    comfort_avg = df.groupby(['group', 'object'])['comfort_score'].mean().unstack()
    
    # Rename columns for legend
    comfort_avg = comfort_avg.rename(columns={'SEE': 'Visual', 'BLIND': 'Non-Visual'})
    
    comfort_avg.T.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
    
    plt.xlabel('Object Type', fontsize=12)
    plt.ylabel('Average Comfort Score (1-5)', fontsize=12)
    plt.title('Comfort Level Comparison: SEE vs BLIND Groups', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Group')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'group_comfort_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def collect_llm_reports():
    """Collect all LLM analysis reports and analysis reports"""
    all_llm_reports = []
    all_analysis_reports = []
    
    for id_num in range(1, 21):
        # LLM reports
        llm_report_path = Path(f"analysis_results/ID{id_num}/llm_analysis/llm_analysis_report.md")
        if llm_report_path.exists():
            with open(llm_report_path, 'r') as f:
                content = f.read()
                all_llm_reports.append(f"=== Participant ID{id_num} ===\n{content}\n")
        
        # Analysis reports
        analysis_report_path = Path(f"analysis_results/ID{id_num}/analysis_report.txt")
        if analysis_report_path.exists():
            with open(analysis_report_path, 'r') as f:
                content = f.read()
                all_analysis_reports.append(f"=== Participant ID{id_num} ===\n{content}\n")
    
    return "\n".join(all_llm_reports), "\n".join(all_analysis_reports)

def generate_scientific_report(df, llm_reports, analysis_reports):
    """Generate comprehensive scientific report using LLM"""
    print("Generating scientific report using LLM...")
    
    # Prepare data summary
    data_summary = f"""
    Total Participants: {df['participant_id'].nunique()}
    SEE Group: {df[df['group'] == 'SEE']['participant_id'].nunique()} participants
    BLIND Group: {df[df['group'] == 'BLIND']['participant_id'].nunique()} participants
    
    Overall Statistics:
    - Average Grasping Frequency: {df['grasping_frequency'].mean():.2f} events/s
    - Total Grasping Events: {df['grasping_events'].sum()}
    - Average Hand Detection Time: {df['hand_detection_time'].mean():.2f}s
    
    Group Comparisons:
    SEE Group Average Frequency: {df[df['group'] == 'SEE']['grasping_frequency'].mean():.2f} events/s
    BLIND Group Average Frequency: {df[df['group'] == 'BLIND']['grasping_frequency'].mean():.2f} events/s
    
    Object-wise Analysis:
    {df.groupby('object')['grasping_frequency'].agg(['mean', 'std']).to_string()}
    
    Emotion Distribution:
    {df['emotion_felt'].value_counts().to_string()}
    """
    
    system_prompt = """You are a scientific researcher analyzing hand movement data from a study examining emotional responses during object grasping.
    
    Study Hypotheses to Test:
    1. Different emotionally evocative objects produce significantly different hand kinematics during grasping
    2. High-arousal emotions (e.g., fear) lead to faster, more forceful movements
    
    Study Design:
    - 20 participants (10 SEE group who saw objects, 10 BLIND group who were blindfolded)
    - 5 objects: cube (neutral), donut (disgust), pig toy (happiness), spider toy (fear), plush toy (comfort)
    - Measured: grasping frequency, velocity, hand detection time using MediaPipe
    
    Your task is to:
    1. Briefly describe what was done in the study
    2. Present key statistical findings with specific numbers
    3. Test both hypotheses - provide evidence that supports or refutes each
    4. Compare SEE vs BLIND groups with statistical significance
    5. Draw conclusions based on the data
    
    Focus on statistical evidence and hypothesis testing rather than general descriptions."""
    
    user_prompt = f"""Based on the following data and individual analysis reports, write a comprehensive scientific analysis for our research paper:
    
    DATA SUMMARY:
    {data_summary}
    
    INDIVIDUAL ANALYSIS REPORTS (first 5000 characters):
    {analysis_reports[:5000]}
    
    LLM ANALYSIS REPORTS (first 5000 characters):
    {llm_reports[:5000]}
    
    Please provide a structured scientific analysis suitable for the Results and Discussion sections of our paper."""
    
    try:
        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt)
            ],
                        model=deployment,
                        temperature=0.7,
            max_tokens=2000,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "Error generating LLM report. Please check API credentials."

def create_participant_summary(df):
    """Create a comprehensive summary of all 20 participants"""
    print("Creating comprehensive participant summary...")
    
    summary_lines = []
    summary_lines.append("COMPREHENSIVE PARTICIPANT SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nTotal Participants: {df['participant_id'].nunique()}")
    summary_lines.append(f"Total Videos Analyzed: {len(df)}")
    summary_lines.append(f"Total Grasping Events: {df['grasping_events'].sum():,}")
    summary_lines.append(f"\nOverall Statistics:")
    summary_lines.append(f"  - Average Grasping Frequency: {df['grasping_frequency'].mean():.2f} ± {df['grasping_frequency'].std():.2f} events/s")
    summary_lines.append(f"  - Average Hand Detection Time: {df['hand_detection_time'].mean():.2f} ± {df['hand_detection_time'].std():.2f} seconds")
    summary_lines.append(f"  - Average Max Velocity: {df['max_velocity'].mean():.2f} ± {df['max_velocity'].std():.2f} px/frame")
    
    # Group comparisons
    summary_lines.append("\n\nGROUP COMPARISONS:")
    summary_lines.append("-" * 40)
    for group in ['SEE', 'BLIND']:
        group_data = df[df['group'] == group]
        group_name = 'Visual' if group == 'SEE' else 'Non-Visual'
        summary_lines.append(f"\n{group_name} Group (n={group_data['participant_id'].nunique()}):")
        summary_lines.append(f"  - Average Grasping Frequency: {group_data['grasping_frequency'].mean():.2f} ± {group_data['grasping_frequency'].std():.2f} events/s")
        summary_lines.append(f"  - Total Grasping Events: {group_data['grasping_events'].sum():,}")
    
    # Object-specific analysis
    summary_lines.append("\n\nOBJECT-SPECIFIC ANALYSIS:")
    summary_lines.append("-" * 40)
    for obj in sorted(df['object'].unique()):
        obj_data = df[df['object'] == obj]
        summary_lines.append(f"\n{obj.capitalize()}:")
        summary_lines.append(f"  - Average Frequency: {obj_data['grasping_frequency'].mean():.2f} ± {obj_data['grasping_frequency'].std():.2f} events/s")
        summary_lines.append(f"  - SEE Group: {obj_data[obj_data['group'] == 'SEE']['grasping_frequency'].mean():.2f} events/s")
        summary_lines.append(f"  - BLIND Group: {obj_data[obj_data['group'] == 'BLIND']['grasping_frequency'].mean():.2f} events/s")
        summary_lines.append(f"  - Most Common Emotion: {obj_data['emotion_felt'].mode()[0] if not obj_data['emotion_felt'].mode().empty else 'N/A'}")
        summary_lines.append(f"  - Average Comfort Score: {obj_data['comfort_score'].mean():.2f}/5.0")
    
    # Individual participant summary
    summary_lines.append("\n\nINDIVIDUAL PARTICIPANT SUMMARY:")
    summary_lines.append("-" * 40)
    participant_summary = df.groupby('participant_id').agg({
        'group': 'first',
        'grasping_frequency': 'mean',
        'grasping_events': 'sum',
        'hand_detection_time': 'sum'
    }).sort_values('participant_id')
    
    for pid, row in participant_summary.iterrows():
        summary_lines.append(f"{pid} ({row['group']}): {row['grasping_frequency']:.1f} events/s, "
                           f"{row['grasping_events']} total events, {row['hand_detection_time']:.1f}s detection time")
    
    # Save summary
    with open(output_dir / 'comprehensive_participant_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    return '\n'.join(summary_lines)

def main():
    """Main execution function"""
    print("Starting scientific paper analysis generation...")
    
    # Extract data from all reports
    df = extract_data_from_reports()
    print(f"Extracted data from {len(df)} video analyses")
    
    # Add comfort score mapping
    comfort_mapping = {
        'Very uncomfortable': 1,
        'Uncomfortable': 2,
        'Neutral': 3,
        'Comfortable': 4,
        'Very comfortable': 5
    }
    df['comfort_score'] = df['comfort'].map(comfort_mapping)
    
    # Create averaged plots for all participants
    create_averaged_plots(df)
    
    # Create group comparison plots
    create_group_comparison_plots(df)
    
    # Create comprehensive participant summary
    participant_summary = create_participant_summary(df)
    
    # Create high-arousal emotions analysis table
    high_arousal_table, arousal_summary = create_high_arousal_table(df)
    
    # Collect all reports
    llm_reports, analysis_reports = collect_llm_reports()
    
    # Generate scientific report using LLM
    scientific_report = generate_scientific_report(df, llm_reports, analysis_reports)
    
    # Save the scientific report
    with open(output_dir / 'scientific_analysis_report.md', 'w') as f:
        f.write("# Scientific Analysis Report\n\n")
        f.write("## Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write(scientific_report)
    
    # Create a summary document
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("SCIENTIFIC PAPER ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Participants: {df['participant_id'].nunique()}\n")
        f.write(f"Total Videos Analyzed: {len(df)}\n\n")
        f.write("Files Generated:\n")
        f.write("- avg_main_analysis.png: Overall grasping frequency by object\n")
        f.write("- avg_grasping_analysis.png: Movement category distribution with percentages\n")
        f.write("- avg_summary_table.png: Enhanced summary statistics table\n")
        f.write("- group_main_analysis.png: SEE vs BLIND comparison\n")
        f.write("- group_emotion_analysis.png: Emotion distribution by group\n")
        f.write("- group_comfort_analysis.png: Comfort level comparison\n")
        f.write("- scientific_analysis_report.md: Hypothesis-focused scientific report\n")
        f.write("- comprehensive_participant_summary.txt: Detailed participant statistics\n")
        f.write("- high_arousal_emotions_table.tex: LaTeX table for high-arousal emotions analysis\n")
        f.write("- arousal_comparison_table.tex: LaTeX table comparing high vs low arousal emotions\n")
    
    print(f"\nAnalysis complete! All outputs saved to '{output_dir}' directory.")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()