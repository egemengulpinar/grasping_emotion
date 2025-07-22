#!/usr/bin/env python3
"""
🔬 SCIENTIFIC HAND MOVEMENT ANALYSIS DEMO
==========================================

Bu script MediaPipe kullanarak el hareketlerinin bilimsel analizini yapar.
Her video için ayrıntılı teknik veriler ve görselleştirmeler üretir.

Kullanım:
- Tek video: python3 demo_hand_analysis.py video.mov
- Klasör (batch): python3 demo_hand_analysis.py --batch video_klasoru/
- Demo veri: python3 demo_hand_analysis.py --demo
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from advanced_hand_analyzer import process_video, batch_process, create_comparison_plots
import argparse

def create_scientific_report(analysis_dir="analysis_results"):
    """
    Bilimsel rapor oluşturur - CSV verilerinden istatistikler çıkarır
    """
    print("📊 Bilimsel Rapor Oluşturuluyor...")
    
    # CSV dosyalarını bul
    csv_files = [f for f in os.listdir(analysis_dir) if f.endswith('_analysis.csv')]
    summary_files = [f for f in os.listdir(analysis_dir) if f.endswith('_summary.json')]
    
    if not csv_files:
        print("❌ Analiz edilmiş video bulunamadı!")
        return
    
    # Bilimsel istatistikler
    all_data = []
    
    for csv_file in csv_files:
        video_name = csv_file.replace('_analysis.csv', '')
        df = pd.read_csv(os.path.join(analysis_dir, csv_file))
        
        # Gelişmiş istatistikler hesapla
        stats = {
            'video': video_name,
            'total_frames': len(df),
            'duration': df['timestamp'].max() - df['timestamp'].min(),
            'max_velocity': df['velocity'].max(),
            'mean_velocity': df['velocity'].mean(),
            'velocity_std': df['velocity'].std(),
            'velocity_cv': df['velocity'].std() / df['velocity'].mean() if df['velocity'].mean() > 0 else 0,
            'max_openness': df['openness'].max(),
            'min_openness': df['openness'].min(),
            'mean_openness': df['openness'].mean(),
            'openness_range': df['openness'].max() - df['openness'].min(),
            'gesture_changes': count_gesture_transitions(df['gesture']),
            'movement_efficiency': calculate_movement_efficiency(df),
            'hand_stability': calculate_stability_index(df),
            'gesture_diversity': len(df['gesture'].unique())
        }
        all_data.append(stats)
    
    # Scientific DataFrame oluştur
    scientific_df = pd.DataFrame(all_data)
    
    # Raporu kaydet
    report_path = os.path.join(analysis_dir, "scientific_report.csv")
    scientific_df.to_csv(report_path, index=False)
    
    # İstatistiksel özet
    print(f"\n📋 BİLİMSEL RAPOR ÖZET ({len(scientific_df)} video)")
    print("="*60)
    print(f"Toplam Analiz Süresi: {scientific_df['duration'].sum():.1f} saniye")
    print(f"Ortalama Maksimum Hız: {scientific_df['max_velocity'].mean():.2f} ± {scientific_df['max_velocity'].std():.2f} px/s")
    print(f"Ortalama El Açıklığı: {scientific_df['mean_openness'].mean():.3f} ± {scientific_df['mean_openness'].std():.3f}")
    print(f"Hız Varyasyon Katsayısı: {scientific_df['velocity_cv'].mean():.3f}")
    print(f"Ortalama Gesture Değişimi: {scientific_df['gesture_changes'].mean():.1f}")
    
    # Korelasyon analizi
    correlation_cols = ['max_velocity', 'mean_velocity', 'mean_openness', 'gesture_changes', 'movement_efficiency']
    correlation_matrix = scientific_df[correlation_cols].corr()
    
    # Korelasyon heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('📊 Bilimsel Metrikler Korelasyon Matrisi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Bilimsel rapor kaydedildi: {report_path}")
    return scientific_df

def count_gesture_transitions(gestures):
    """Gesture geçişlerini sayar"""
    transitions = 0
    for i in range(1, len(gestures)):
        if gestures.iloc[i] != gestures.iloc[i-1]:
            transitions += 1
    return transitions

def calculate_movement_efficiency(df):
    """Hareket verimliliği hesaplar (düz çizgi mesafesi / gerçek yol)"""
    if len(df) < 2:
        return 1.0
    
    # Başlangıç ve bitiş noktaları
    start_x, start_y = df['center_x'].iloc[0], df['center_y'].iloc[0]
    end_x, end_y = df['center_x'].iloc[-1], df['center_y'].iloc[-1]
    
    # Düz çizgi mesafesi
    straight_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # Gerçek yol uzunluğu
    actual_path = 0
    for i in range(1, len(df)):
        if pd.notna(df['center_x'].iloc[i]) and pd.notna(df['center_x'].iloc[i-1]):
            dx = df['center_x'].iloc[i] - df['center_x'].iloc[i-1]
            dy = df['center_y'].iloc[i] - df['center_y'].iloc[i-1]
            actual_path += np.sqrt(dx**2 + dy**2)
    
    return straight_distance / actual_path if actual_path > 0 else 1.0

def calculate_stability_index(df):
    """El kararlılığı indeksi hesaplar (düşük hız varyasyonu = yüksek kararlılık)"""
    if len(df) < 3:
        return 1.0
    
    # Hız değişimlerinin standart sapması
    velocity_changes = df['velocity'].diff().dropna()
    stability = 1 / (1 + velocity_changes.std()) if velocity_changes.std() > 0 else 1.0
    
    return min(stability, 1.0)

def create_advanced_visualizations(analysis_dir="analysis_results"):
    """
    Gelişmiş bilimsel görselleştirmeler oluşturur
    """
    print("📈 Gelişmiş Görselleştirmeler Oluşturuluyor...")
    
    csv_files = [f for f in os.listdir(analysis_dir) if f.endswith('_analysis.csv')]
    
    if not csv_files:
        print("❌ Analiz verisi bulunamadı!")
        return
    
    # Set style for scientific plots
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Figure 1: Multi-video velocity comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔬 Multi-Video Scientific Analysis Dashboard', fontsize=18, fontweight='bold')
    
    all_velocities = []
    all_openness = []
    video_names = []
    
    for i, csv_file in enumerate(csv_files[:4]):  # Max 4 videos for clarity
        video_name = csv_file.replace('_analysis.csv', '')
        df = pd.read_csv(os.path.join(analysis_dir, csv_file))
        
        # Velocity time series
        axes[0,0].plot(df['timestamp'], df['velocity'], label=video_name, alpha=0.8, linewidth=1.5)
        
        # Openness time series  
        axes[0,1].plot(df['timestamp'], df['openness'], label=video_name, alpha=0.8, linewidth=1.5)
        
        all_velocities.extend(df['velocity'].dropna())
        all_openness.extend(df['openness'].dropna())
        video_names.append(video_name)
    
    axes[0,0].set_title('Velocity Time Series Comparison', fontweight='bold')
    axes[0,0].set_xlabel('Time (seconds)')
    axes[0,0].set_ylabel('Velocity (px/s)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title('Hand Openness Time Series', fontweight='bold')
    axes[0,1].set_xlabel('Time (seconds)')
    axes[0,1].set_ylabel('Openness (0-1)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Velocity distribution
    axes[1,0].hist(all_velocities, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(np.mean(all_velocities), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(all_velocities):.1f}')
    axes[1,0].axvline(np.median(all_velocities), color='green', linestyle='--',
                     label=f'Median: {np.median(all_velocities):.1f}')
    axes[1,0].set_title('Velocity Distribution (All Videos)', fontweight='bold')
    axes[1,0].set_xlabel('Velocity (px/s)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Openness distribution
    axes[1,1].hist(all_openness, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1,1].axvline(np.mean(all_openness), color='red', linestyle='--',
                     label=f'Mean: {np.mean(all_openness):.3f}')
    axes[1,1].set_title('Hand Openness Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Openness (0-1)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "advanced_scientific_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gelişmiş görselleştirmeler oluşturuldu!")

def generate_summary_statistics(analysis_dir="analysis_results"):
    """
    Özet istatistikleri oluşturur ve yazdırır
    """
    summary_files = [f for f in os.listdir(analysis_dir) if f.endswith('_summary.json')]
    
    if not summary_files:
        print("❌ Özet dosyaları bulunamadı!")
        return
    
    print("\n🎯 DETAYLI İSTATİSTİKLER")
    print("="*50)
    
    for summary_file in summary_files:
        with open(os.path.join(analysis_dir, summary_file), 'r') as f:
            data = json.load(f)
        
        video_name = summary_file.replace('_summary.json', '')
        print(f"\n📹 {video_name.upper()}")
        print("-" * 30)
        print(f"Süre: {data['duration']:.2f} saniye")
        print(f"Toplam Frame: {data['total_frames']:,}")
        print(f"Maksimum Hız: {data['max_velocity']:.2f} px/s")
        print(f"Ortalama Hız: {data['avg_velocity']:.2f} px/s")
        print(f"Toplam Yol: {data['path_length']:.2f} piksel")
        print(f"En Yaygın Gesture: {data['most_common_gesture']}")
        print(f"Ortalama El Açıklığı: {data['avg_openness']:.3f}")

def run_demo():
    """
    Demo modunu çalıştırır - mevcut videolar ile örnek analiz
    """
    print("🚀 DEMO MODU - Mevcut videolar analiz ediliyor...")
    
    # Mevcut video dosyalarını bul
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    current_dir = os.getcwd()
    
    video_files = []
    for ext in video_extensions:
        video_files.extend([f for f in os.listdir(current_dir) if f.lower().endswith(ext)])
    
    if not video_files:
        print("❌ Video dosyası bulunamadı!")
        print("Lütfen .mp4, .mov, .avi veya .mkv dosyası ekleyin.")
        return
    
    print(f"📹 {len(video_files)} video dosyası bulundu: {', '.join(video_files)}")
    
    # İlk videoyu analiz et
    if len(video_files) >= 1:
        print(f"\n🎬 Analiz ediliyor: {video_files[0]}")
        summary = process_video(video_files[0], "demo_results")
        
        if summary:
            print("\n✅ DEMO ANALİZ TAMAMLANDI!")
            print(f"Sonuçlar 'demo_results' klasöründe")
            
            # Bilimsel rapor oluştur
            create_scientific_report("demo_results")
            create_advanced_visualizations("demo_results")
            generate_summary_statistics("demo_results")

def main():
    parser = argparse.ArgumentParser(description='🔬 Bilimsel El Hareketi Analizi')
    parser.add_argument('input', nargs='?', help='Video dosyası veya klasör yolu')
    parser.add_argument('--batch', type=str, help='Klasördeki tüm videoları analiz et')
    parser.add_argument('--demo', action='store_true', help='Demo modunu çalıştır')
    parser.add_argument('--output', type=str, default='scientific_analysis', help='Çıktı klasörü')
    parser.add_argument('--report', action='store_true', help='Sadece rapor oluştur (var olan verilerden)')
    
    args = parser.parse_args()
    
    print("🔬 BİLİMSEL EL HAREKETİ ANALİZ SİSTEMİ")
    print("="*50)
    
    if args.demo:
        run_demo()
        
    elif args.report:
        print("📊 Mevcut verilerden rapor oluşturuluyor...")
        create_scientific_report(args.output)
        create_advanced_visualizations(args.output)
        generate_summary_statistics(args.output)
        
    elif args.batch:
        print(f"🎯 Batch analiz başlatılıyor: {args.batch}")
        summaries = batch_process(args.batch, args.output)
        
        if summaries:
            print(f"\n🎉 {len(summaries)} video başarıyla analiz edildi!")
            create_scientific_report(args.output)
            create_advanced_visualizations(args.output)
            generate_summary_statistics(args.output)
            
    elif args.input:
        print(f"🎬 Tek video analiz: {args.input}")
        summary = process_video(args.input, args.output)
        
        if summary:
            print("\n✅ Analiz tamamlandı!")
            generate_summary_statistics(args.output)
            
    else:
        print("❌ Lütfen bir input belirtin:")
        print("  Demo için: python3 demo_hand_analysis.py --demo")
        print("  Tek video: python3 demo_hand_analysis.py video.mov")
        print("  Batch: python3 demo_hand_analysis.py --batch video_klasoru/")

if __name__ == "__main__":
    main() 