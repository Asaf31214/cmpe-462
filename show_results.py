#!/usr/bin/env python3
"""
Results visualization and summary generation for CMPE-462 Assignment 1
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_file: str) -> Dict[str, Any]:
    """Load results JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_comparison_table(results: Dict[str, Any]) -> str:
    """Generate formatted comparison table."""
    table = []
    table.append("=" * 120)
    table.append("COMPREHENSIVE RESULTS COMPARISON: SCRATCH vs SCIKIT-LEARN")
    table.append("=" * 120)
    
    modalities = ['image', 'num_cat', 'text', 'fused']
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    for modality in modalities:
        if modality not in results:
            continue
        
        table.append(f"\n{'─' * 120}")
        table.append(f"MODALITY: {modality.upper()}")
        table.append(f"{'─' * 120}")
        
        scratch_metrics = results[modality]['scratch']['metrics']
        sklearn_metrics = results[modality]['sklearn']['metrics']
        
        scratch_time = results[modality]['scratch']['train_time']
        sklearn_time = results[modality]['sklearn']['train_time']
        
        # Header
        table.append(f"{'Metric':<20} {'Scratch':<18} {'Sklearn':<18} {'Difference':<18} {'Match':<8}")
        table.append("-" * 120)
        
        # Metrics
        for metric in metrics:
            scratch_val = scratch_metrics.get(metric, 0)
            sklearn_val = sklearn_metrics.get(metric, 0)
            diff = abs(sklearn_val - scratch_val)
            match = "✓" if diff < 0.01 else "~" if diff < 0.05 else "✗"
            
            metric_name = metric.replace('_', ' ').title()
            table.append(f"{metric_name:<20} {scratch_val:<18.6f} {sklearn_val:<18.6f} {diff:>+18.6f} {match:>8}")
        
        table.append(f"{'Train Time (s)':<20} {scratch_time:<18.6f} {sklearn_time:<18.6f} {scratch_time - sklearn_time:>+18.6f}")
        
        # Feature count
        feature_counts = {
            'image': 70,
            'num_cat': 11,
            'text': 100,
            'fused': 181
        }
        table.append(f"{'Features':<20} {feature_counts[modality]:<18} {'-':<18}")
    
    return "\n".join(table)

def generate_summary_stats(results: Dict[str, Any]) -> str:
    """Generate summary statistics."""
    stats = []
    stats.append("\n" + "=" * 120)
    stats.append("SUMMARY STATISTICS")
    stats.append("=" * 120)
    
    modalities = ['image', 'num_cat', 'text', 'fused']
    
    max_acc_mod = None
    max_acc = 0
    min_time_scratch = float('inf')
    max_time_scratch = 0
    
    total_time_scratch = 0
    total_time_sklearn = 0
    
    for modality in modalities:
        if modality not in results:
            continue
        
        scratch_acc = results[modality]['scratch']['metrics']['accuracy']
        sklearn_acc = results[modality]['sklearn']['metrics']['accuracy']
        scratch_time = results[modality]['scratch']['train_time']
        sklearn_time = results[modality]['sklearn']['train_time']
        
        if scratch_acc > max_acc:
            max_acc = scratch_acc
            max_acc_mod = modality
        
        min_time_scratch = min(min_time_scratch, scratch_time)
        max_time_scratch = max(max_time_scratch, scratch_time)
        total_time_scratch += scratch_time
        total_time_sklearn += sklearn_time
    
    stats.append(f"\nBest performing modality (scratch): {max_acc_mod.upper()} with {max_acc*100:.2f}% accuracy")
    stats.append(f"Fastest modality: {min_time_scratch*1000:.1f} ms")
    stats.append(f"Slowest modality: {max_time_scratch*1000:.1f} ms")
    stats.append(f"\nTotal training time (scratch): {total_time_scratch:.3f}s")
    stats.append(f"Total training time (sklearn): {total_time_sklearn:.3f}s")
    stats.append(f"Speedup ratio (sklearn/scratch): {total_time_scratch/total_time_sklearn:.1f}x")
    
    return "\n".join(stats)

def main():
    """Main function."""
    results = load_results('results/summary.json')
    
    # Generate output
    comparison = generate_comparison_table(results)
    stats = generate_summary_stats(results)
    
    # Print to console
    print(comparison)
    print(stats)
    
    # Save to file
    with open('DETAILED_RESULTS.txt', 'w') as f:
        f.write(comparison)
        f.write(stats)
    
    print(f"\n✓ Results saved to DETAILED_RESULTS.txt")
    
    # Generate plots summary
    print("\n" + "=" * 120)
    print("LOSS CONVERGENCE PLOTS GENERATED")
    print("=" * 120)
    
    plot_files = [
        'results/loss_image.png',
        'results/loss_num_cat.png',
        'results/loss_text.png',
        'results/loss_fused.png'
    ]
    
    for plot_file in plot_files:
        if Path(plot_file).exists():
            size_kb = Path(plot_file).stat().st_size / 1024
            print(f"✓ {plot_file:<35} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    main()
