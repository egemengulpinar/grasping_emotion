import pandas as pd
import json
import os
from scipy import stats

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
                
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                participant_velocities[f"ID{participant_id}"] = 0.0
        else:
            participant_velocities[f"ID{participant_id}"] = 0.0
    
    return participant_velocities

def load_and_prepare_data_from_llm_input():
    """
    Load data from the llm_analysis_input.json file, process it into a DataFrame,
    and add experimental group and arousal information. Now includes real velocity data from CSV files.
    """
    file_path = 'analysis_results/llm_analysis_input.json'
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run the main analysis script first.")
        return pd.DataFrame()

    # Extract velocity data from CSV files
    participant_velocities = extract_velocity_from_csv_files()

    # The actual data is under the 'participant_data' key
    trial_data = raw_data.get('participant_data', [])
    if not trial_data:
        print("Error: 'participant_data' key not found or is empty in the JSON file.")
        return pd.DataFrame()

    data = []
    for trial in trial_data:
        # The group is already in the data as 'SEE' or 'BLIND'
        group = 'Visual' if trial.get('group') == 'SEE' else 'Non-Visual'

        # Define arousal categories based on the paper
        emotion = trial.get('emotion', 'Neutral')
        high_arousal_emotions = ['Fear', 'Disgust', 'Surprise']
        arousal_category = 'High-Arousal' if emotion in high_arousal_emotions else 'Low-Arousal'

        # Get real velocity data from CSV analysis
        participant_id = trial['participant_id']
        real_velocity = participant_velocities.get(participant_id, 0.0)

        data.append({
            'participant_id': trial['participant_id'],
            'group': group,
            'object': trial['object_type'], # 'object_type' is the correct key
            'grasping_frequency': trial.get('grasping_frequency', 0),
            'velocity': real_velocity, # Use real velocity from CSV files
            'emotion': emotion,
            'arousal_category': arousal_category
        })
            
    df = pd.DataFrame(data)
    return df

def run_analyses(df):
    """
    Run all required statistical tests and print the results.
    """
    print("--- Statistical Analysis Results ---")
    
    # --- Analysis 1: Group Comparison (Visual vs. Non-Visual) ---
    print("\n1. Visual vs. Non-Visual Group Comparison (Grasping Frequency)")
    visual_group = df[df['group'] == 'Visual']['grasping_frequency']
    non_visual_group = df[df['group'] == 'Non-Visual']['grasping_frequency']
    
    # Check if groups have data
    if not visual_group.empty and not non_visual_group.empty:
        u_stat_group, p_val_group = stats.mannwhitneyu(visual_group, non_visual_group, alternative='two-sided')
        print(f"   - Mann-Whitney U test: U = {u_stat_group:.1f}, p = {p_val_group:.3f}")
        # Also print means for context
        print(f"   - Mean (Visual): {visual_group.mean():.2f}, Mean (Non-Visual): {non_visual_group.mean():.2f}")
    else:
        print("   - Not enough data for group comparison.")

    # --- Analysis 2: Object-Specific Kinematic Patterns (Hypothesis 1) ---
    print("\n2. Object Comparison (Grasping Frequency)")
    df['object'] = df['object'].str.strip().str.lower()
    objects = sorted(df['object'].unique()) # Sort for consistent order
    
    if len(objects) > 1:
        object_data = [df[df['object'] == obj]['grasping_frequency'] for obj in objects]
        
        if all(len(d) > 0 for d in object_data):
            h_stat_obj, p_val_obj = stats.kruskal(*object_data)
            df_kruskal = len(objects) - 1
            print(f"   - Kruskal-Wallis H test: H({df_kruskal}) = {h_stat_obj:.1f}, p = {p_val_obj:.3f}")
            print(f"   - Object means:")
            for obj in objects:
                mean_freq = df[df['object'] == obj]['grasping_frequency'].mean()
                print(f"     - {obj.capitalize()}: {mean_freq:.2f}")

        else:
            print("   - Not enough data for Kruskal-Wallis test.")
    else:
        print(f"   - Only one object type found ('{objects[0]}'), skipping Kruskal-Wallis test.")


    # --- Analysis 3: Arousal-Based Kinematic Analysis (Hypothesis 2) ---
    print("\n3. Arousal-Based Comparison")
    high_arousal_df = df[df['arousal_category'] == 'High-Arousal']
    low_arousal_df = df[df['arousal_category'] == 'Low-Arousal']

    # 3a: Grasping Frequency
    print("   a) Grasping Frequency")
    if not high_arousal_df.empty and not low_arousal_df.empty:
        u_stat_freq, p_val_freq = stats.mannwhitneyu(high_arousal_df['grasping_frequency'], low_arousal_df['grasping_frequency'], alternative='two-sided')
        print(f"      - Mann-Whitney U test: U = {u_stat_freq:.1f}, p = {p_val_freq:.3f}")
        print(f"      - Mean (High): {high_arousal_df['grasping_frequency'].mean():.2f}, Mean (Low): {low_arousal_df['grasping_frequency'].mean():.2f}")
        print(f"      - Std (High): {high_arousal_df['grasping_frequency'].std():.2f}, Std (Low): {low_arousal_df['grasping_frequency'].std():.2f}")
        print(f"      - N (High): {len(high_arousal_df)}, N (Low): {len(low_arousal_df)}")
    else:
        print("      - Not enough data for arousal frequency comparison.")
        
    # 3b: Velocity (Average Velocity from CSV analysis)
    print("   b) Velocity (Average velocity in px/frame from CSV files)")
    if not high_arousal_df.empty and not low_arousal_df.empty:
        # Ensure there's velocity data to compare
        if 'velocity' in high_arousal_df.columns and 'velocity' in low_arousal_df.columns:
            u_stat_vel, p_val_vel = stats.mannwhitneyu(high_arousal_df['velocity'], low_arousal_df['velocity'], alternative='two-sided')
            print(f"      - Mann-Whitney U test: U = {u_stat_vel:.1f}, p = {p_val_vel:.3f}")
            print(f"      - Mean (High): {high_arousal_df['velocity'].mean():.2f}, Mean (Low): {low_arousal_df['velocity'].mean():.2f}")
            print(f"      - Std (High): {high_arousal_df['velocity'].std():.2f}, Std (Low): {low_arousal_df['velocity'].std():.2f}")
            print(f"      - N (High): {len(high_arousal_df)}, N (Low): {len(low_arousal_df)}")
        else:
            print("      - Velocity data not found in DataFrame.")
    else:
        print("      - Not enough data for arousal velocity comparison.")
        
    print("\n--- End of Analysis ---")


if __name__ == '__main__':
    master_df = load_and_prepare_data_from_llm_input()
    
    if not master_df.empty:
        print("Data loaded successfully.")
        print(f"Total records processed: {len(master_df)}")
        print("\nRunning statistical analyses...")
        run_analyses(master_df)
    else:
        print("Failed to load or process data. The DataFrame is empty.") 