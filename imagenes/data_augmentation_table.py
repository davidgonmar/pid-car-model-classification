import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_data_augmentation_table():
    """Create a visual representation of data augmentation impact on model performance"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set background color
    ax.set_facecolor('#f8f8f8')
    
    # Define augmentation techniques and metrics
    techniques = [
        'Original Dataset (No Aug.)',
        'Random Horizontal Flip',
        'Random Rotation (±15°)',
        'Color Jitter',
        'Random Brightness',
        'Random Contrast',
        'Random Crop',
        'Gaussian Noise',
        'Gaussian Blur',
        'All Combined'
    ]
    
    metrics = [
        'Training Time (rel.)',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'Generalization',
        'Robustness'
    ]
    
    # Simulated data representing the impact of each augmentation technique
    # Values range from -3 (very negative impact) to +3 (very positive impact)
    # 0 represents neutral/baseline
    impact_data = np.array([
        # Training Time (negative means longer), Accuracy, Precision, Recall, F1, Generalization, Robustness
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # Original dataset (baseline)
        [0.1,  1.2,  1.0,  1.5,  1.2,  1.8,  1.5],  # Random Horizontal Flip
        [-0.2, 1.0,  0.8,  1.2,  1.0,  1.7,  1.3],  # Random Rotation
        [-0.4, 1.3,  1.1,  1.4,  1.3,  2.0,  1.9],  # Color Jitter
        [-0.1, 0.7,  0.6,  0.8,  0.7,  1.5,  1.6],  # Random Brightness
        [-0.2, 0.9,  0.8,  1.0,  0.9,  1.4,  1.5],  # Random Contrast
        [-0.3, 1.5,  1.3,  1.6,  1.5,  1.9,  1.7],  # Random Crop
        [-0.5, 0.6,  0.5,  0.7,  0.6,  1.1,  2.2],  # Gaussian Noise
        [-0.4, 0.3,  0.2,  0.4,  0.3,  0.9,  1.8],  # Gaussian Blur
        [-1.5, 2.5,  2.3,  2.6,  2.5,  2.8,  2.9],  # All Combined
    ])
    
    # Create a custom colormap: red for negative, white for neutral, green for positive
    cmap = LinearSegmentedColormap.from_list('custom_RdWhGn', ['#d9534f', '#f9f9f9', '#5cb85c'])
    
    # Create a DataFrame for easier plotting with seaborn
    df = pd.DataFrame(impact_data, index=techniques, columns=metrics)
    
    # Create heatmap
    sns.heatmap(df, cmap=cmap, annot=True, fmt=".1f", linewidths=.5, 
                cbar_kws={'label': 'Impact Score (-3 to +3)'}, vmin=-3, vmax=3,
                annot_kws={"size": 10, "weight": "bold"}, ax=ax)
    
    # Format the plot
    ax.set_title('Impact of Data Augmentation Techniques on Model Performance', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add explanatory annotations
    explanation = """
    Impact Score Interpretation:
    • Positive values indicate improvement over baseline
    • Negative values indicate degradation compared to baseline
    • 0 represents the baseline (no change)
    
    Note: For Training Time, negative values represent longer times (worse),
    while positive values indicate shorter times (better).
    """
    
    plt.figtext(1.1, 0.1, explanation, ha='center', fontsize=11, 
                bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=1'))
    
    # Add annotations for key findings
    key_findings = [
        "Key Findings:",
        "1. Combined augmentations provide the best overall performance boost",
        "2. Horizontal flips and random crops offer the best performance-to-time ratio",
        "3. Blur and noise significantly increase robustness despite limited accuracy gains",
        "4. All augmentation techniques improve generalization"
    ]
    
    findings_text = '\n'.join(key_findings)
    plt.figtext(0.85, 0.85, findings_text, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='#e9ecef', edgecolor='gray', boxstyle='round,pad=1'))
    
    # Add an outline of specific augmentation implementations
    implementation = """
    Implementation Details:
    • Horizontal Flip: p=0.5
    • Rotation: ±15° with p=0.3
    • Color Jitter: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    • Crop: 224×224 from 256×256 images
    • Noise: Gaussian σ=0.01
    • Blur: Gaussian kernel=3×3
    """
    
    plt.figtext(0.95, 0.4, implementation, fontsize=11,
                bbox=dict(facecolor='#e9ecef', edgecolor='gray', boxstyle='round,pad=1'))
    
    # Create secondary bar chart showing absolute metrics
    # Add a new axes for the bar chart
    ax2 = fig.add_axes([0.1, 0.02, 0.8, 0.15])
    
    # Absolute metrics for select methods (baseline, best single augmentation, all combined)
    methods = ['No Augmentation', 'Random Crop', 'All Combined']
    accuracy_values = [82.3, 87.1, 91.5]
    f1_values = [81.9, 86.5, 90.8]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax2.bar(x - width/2, accuracy_values, width, label='Accuracy (%)', color='#5cb85c', edgecolor='black')
    ax2.bar(x + width/2, f1_values, width, label='F1 Score (%)', color='#5bc0de', edgecolor='black')
    
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Absolute Metrics Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(accuracy_values):
        ax2.text(i - width/2, v + 0.5, f"{v}%", ha='center', fontweight='bold')
    
    for i, v in enumerate(f1_values):
        ax2.text(i + width/2, v + 0.5, f"{v}%", ha='center', fontweight='bold')
    
    # Adjust the main plot to make room for the secondary plot
    plt.subplots_adjust(bottom=0.25)
    
    # Save figure
    plt.savefig('data_augmentation_impact.jpg', dpi=300, bbox_inches='tight')
    print("Data augmentation impact chart saved to 'data_augmentation_impact.jpg'")

if __name__ == "__main__":
    create_data_augmentation_table() 