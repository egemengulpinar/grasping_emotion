import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem # To calculate Standard Error of the Mean
import os

def extract_velocity_from_csv_files():
    """Extract velocity data from detailed_analysis.csv files for all participants"""
    participant_velocities = {}
    
    for participant_id in range(1, 21):
        csv_path = f"analysis_results/ID{participant_id}/detailed_analysis.csv"
        
        if os.path.exists(csv_path):
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
                
                # Calculate mean velocity for this participant across all frames
                mean_velocity = df['velocity'].mean()
                participant_velocities[f"ID{participant_id}"] = mean_velocity
                
                print(f"ID{participant_id}: Mean velocity = {mean_velocity:.2f} px/frame")
                
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                participant_velocities[f"ID{participant_id}"] = 0.0
        else:
            print(f"Warning: {csv_path} not found")
            participant_velocities[f"ID{participant_id}"] = 0.0
    
    return participant_velocities

def load_arousal_data():
    """Load arousal categorization data"""
    file_path = 'analysis_results/llm_analysis_input.json'
    
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}

    participant_data = raw_data.get('participant_data', [])
    
    # Group by participant and get their predominant emotion/arousal
    participant_emotions = {}
    for trial in participant_data:
        pid = trial['participant_id']
        emotion = trial.get('emotion', 'Neutral')
        
        if pid not in participant_emotions:
            participant_emotions[pid] = []
        participant_emotions[pid].append(emotion)
    
    # Determine arousal category for each participant based on most common emotion
    high_arousal_emotions = ['Fear', 'Disgust', 'Surprise']
    participant_arousal = {}
    
    for pid, emotions in participant_emotions.items():
        # Count emotions and find the most common
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        arousal_category = 'High-Arousal' if most_common_emotion in high_arousal_emotions else 'Low-Arousal'
        participant_arousal[pid] = arousal_category
        
        print(f"{pid}: Most common emotion = {most_common_emotion}, Arousal = {arousal_category}")
    
    return participant_arousal

def create_hypothesis2_plot():
    """Create the Hypothesis 2 comparison plot with correct velocity data from CSV files"""
    
    print("🔍 Extracting velocity data from detailed_analysis.csv files...")
    participant_velocities = extract_velocity_from_csv_files()
    
    print("\n🔍 Loading arousal categorization data...")
    participant_arousal = load_arousal_data()
    
    # Load frequency data from existing JSON
    file_path = 'analysis_results/llm_analysis_input.json'
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    trial_data = raw_data.get('participant_data', [])
    
    # Group data by arousal category for frequency
    high_arousal_data = {'frequencies': [], 'velocities': []}
    low_arousal_data = {'frequencies': [], 'velocities': []}
    
    high_arousal_emotions = ['Fear', 'Disgust', 'Surprise']
    
    for trial in trial_data:
        emotion = trial.get('emotion', 'Neutral')
        arousal_category = 'High-Arousal' if emotion in high_arousal_emotions else 'Low-Arousal'
        
        participant_id = trial['participant_id']
        frequency = trial.get('grasping_frequency', 0)
        
        # Get velocity from our CSV extraction
        velocity = participant_velocities.get(participant_id, 0.0)
        
        if arousal_category == 'High-Arousal':
            high_arousal_data['frequencies'].append(frequency)
            high_arousal_data['velocities'].append(velocity)
        else:
            low_arousal_data['frequencies'].append(frequency)
            low_arousal_data['velocities'].append(velocity)
    
    # Calculate statistics
    high_freq_mean = np.mean(high_arousal_data['frequencies'])
    high_freq_std = np.std(high_arousal_data['frequencies'])
    high_vel_mean = np.mean(high_arousal_data['velocities'])
    high_vel_std = np.std(high_arousal_data['velocities'])
    high_n = len(high_arousal_data['frequencies'])
    
    low_freq_mean = np.mean(low_arousal_data['frequencies'])
    low_freq_std = np.std(low_arousal_data['frequencies'])
    low_vel_mean = np.mean(low_arousal_data['velocities'])
    low_vel_std = np.std(low_arousal_data['velocities'])
    low_n = len(low_arousal_data['frequencies'])
    
    print(f"\n📊 Statistics:")
    print(f"High-Arousal (n={high_n}): Freq = {high_freq_mean:.2f}±{high_freq_std:.2f}, Vel = {high_vel_mean:.2f}±{high_vel_std:.2f}")
    print(f"Low-Arousal (n={low_n}): Freq = {low_freq_mean:.2f}±{low_freq_std:.2f}, Vel = {low_vel_mean:.2f}±{low_vel_std:.2f}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors to match your existing figures
    colors = ['#48D1CC', '#FA8072']  # Teal and Salmon
    
    # Plot 1: Grasping Frequency
    categories = ['High-Arousal\nEmotions', 'Low-Arousal\nEmotions']
    freq_means = [high_freq_mean, low_freq_mean]
    freq_stds = [high_freq_std, low_freq_std]
    
    bars1 = ax1.bar(categories, freq_means, yerr=freq_stds, 
                   color=colors, alpha=0.8, capsize=8, 
                   edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Grasping Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Grasping Frequency Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(freq_means, freq_stds)):
        ax1.text(i, mean_val + std_val + 0.5, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Movement Velocity
    vel_means = [high_vel_mean, low_vel_mean]
    vel_stds = [high_vel_std, low_vel_std]
    
    bars2 = ax2.bar(categories, vel_means, yerr=vel_stds, 
                   color=colors, alpha=0.8, capsize=8, 
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Average Velocity (px/frame)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Movement Velocity Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(vel_means, vel_stds)):
        ax2.text(i, mean_val + std_val + 0.02, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add sample sizes to legend
    legend_labels = [f'High-Arousal (N={high_n})', f'Low-Arousal (N={low_n})']
    fig.legend(bars1, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=2, fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Overall title
    plt.suptitle('Hypothesis 2: Arousal Level vs Hand Movement Patterns\n' + 
                '(Error bars represent ± 1 Standard Deviation)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    output_path = 'Final_Report/figures/hypothesis2_arousal_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    create_hypothesis2_plot() 