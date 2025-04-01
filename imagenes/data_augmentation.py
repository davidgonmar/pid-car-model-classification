import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def create_test_image(size=224):
    """Create a simple car image for demonstration"""
    img = np.ones((size, size, 3))
    
    # Car body
    img[50:150, 60:180, :] = [0.7, 0.7, 0.7]
    
    # Windows
    img[30:50, 90:160, :] = [0.3, 0.3, 0.3] 
    
    # Wheels
    y, x = np.ogrid[0:size, 0:size]
    front_wheel = (x - 80)**2 + (y - 150)**2 <= 15**2
    back_wheel = (x - 160)**2 + (y - 150)**2 <= 15**2
    img[front_wheel] = [0.1, 0.1, 0.1]
    img[back_wheel] = [0.1, 0.1, 0.1]
    
    # Lights
    img[90:105, 60:75, :] = [1.0, 0.8, 0.2]  # front light
    img[90:105, 165:180, :] = [0.9, 0.2, 0.2]  # back light
    
    return img

def apply_horizontal_flip(img):
    """Apply horizontal flip augmentation"""
    return img[:, ::-1, :]

def apply_rotation(img, angle=10):
    """Apply rotation augmentation"""
    from scipy.ndimage import rotate
    rotated = rotate(img, angle, reshape=False)
    rotated = np.clip(rotated, 0, 1)  # Clip values to valid range
    return rotated

def apply_color_jitter(img):
    """Apply color jitter augmentation"""
    brightness = 0.2
    contrast = 0.2
    saturation = 0.2
    
    # Adjust brightness
    result = img * (1 + brightness * (np.random.random() - 0.5))
    
    # Adjust contrast
    gray = np.mean(result, axis=2, keepdims=True)
    result = (result - gray) * (1 + contrast * (np.random.random() - 0.5)) + gray
    
    # Adjust saturation (simplified)
    gray = np.mean(result, axis=2, keepdims=True)
    result = result + (gray - result) * (saturation * (np.random.random() - 0.5))
    
    # Clip values to valid range
    result = np.clip(result, 0, 1)
    
    return result

def apply_random_crop(img, size=224, crop_size=190):
    """Apply random crop augmentation"""
    result = np.ones((size, size, 3))
    
    # Calculate random offset (but not too far)
    max_offset = size - crop_size
    offset_x = np.random.randint(0, max_offset)
    offset_y = np.random.randint(0, max_offset)
    
    # Crop and resize
    cropped = img[offset_y:offset_y+crop_size, offset_x:offset_x+crop_size, :]
    
    # Place back in center of full-sized image
    pad_x = (size - crop_size) // 2
    pad_y = (size - crop_size) // 2
    result[pad_y:pad_y+crop_size, pad_x:pad_x+crop_size, :] = cropped
    
    return result

def apply_random_erasing(img, size=224):
    """Apply random erasing augmentation"""
    result = img.copy()
    
    # Random rectangle to erase
    erase_x = np.random.randint(50, 150)
    erase_y = np.random.randint(50, 150)
    erase_w = np.random.randint(20, 60)
    erase_h = np.random.randint(20, 60)
    
    # Make sure we don't go out of bounds
    erase_x = min(erase_x, size - erase_w)
    erase_y = min(erase_y, size - erase_h)
    
    # Fill with random color or gray
    if np.random.random() > 0.5:
        color = np.random.random(3)
    else:
        color = [0.5, 0.5, 0.5]
    
    result[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w, :] = color
    
    return result

def plot_augmentation_examples():
    """Plot examples of different augmentation techniques"""
    # Create base image
    base_img = create_test_image()
    
    # Apply augmentations
    flipped_img = apply_horizontal_flip(base_img)
    rotated_img = apply_rotation(base_img)
    color_img = apply_color_jitter(base_img)
    cropped_img = apply_random_crop(base_img)
    erased_img = apply_random_erasing(base_img)
    
    # Create combined sample with multiple augmentations
    combined_img = base_img.copy()
    combined_img = apply_horizontal_flip(combined_img)
    combined_img = apply_color_jitter(combined_img)
    combined_img = apply_random_erasing(combined_img)
    
    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(base_img)
    ax0.set_title("Original", fontsize=12)
    ax0.axis('off')
    
    # Augmented images
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(flipped_img)
    ax1.set_title("RandomHorizontalFlip", fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(rotated_img)
    ax2.set_title("RandomRotation", fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(color_img)
    ax3.set_title("ColorJitter", fontsize=12)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cropped_img)
    ax4.set_title("RandomCrop", fontsize=12)
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(erased_img)
    ax5.set_title("RandomErasing", fontsize=12)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[2, :])
    ax6.imshow(combined_img)
    ax6.set_title("Combinación de Múltiples Técnicas", fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.suptitle("Técnicas de Data Augmentation para Clasificación de Vehículos", 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "La combinación de técnicas de data augmentation mejora la robustez del modelo\n"
               "frente a variaciones en iluminación, perspectiva y oclusiones parciales.",
               ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.savefig('data_augmentation_examples.jpg', dpi=300, bbox_inches='tight')
    print("Data augmentation examples saved to 'data_augmentation_examples.jpg'")

def plot_augmentation_impact():
    """Plot the impact of data augmentation techniques on model performance"""
    # Performance data (from the paper)
    techniques = ['Baseline', 'RandomHorizontalFlip', 'RandomRotation', 
                 'ColorJitter', 'RandomCrop', 'RandomErasing', 'Combinación']
    
    precision = [84.9, 87.0, 86.7, 87.5, 86.1, 86.8, 89.2]
    robustness = [62.3, 67.6, 69.4, 72.0, 65.8, 70.7, 77.0]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar positions
    x = np.arange(len(techniques))
    width = 0.35
    
    # Create bars
    precision_bars = ax.bar(x - width/2, precision, width, label='Precisión (%)', 
                           color='royalblue', alpha=0.8)
    robustness_bars = ax.bar(x + width/2, robustness, width, label='Robustez (%)', 
                            color='firebrick', alpha=0.8)
    
    # Customize chart
    ax.set_ylabel('Porcentaje (%)', fontsize=12)
    ax.set_title('Impacto de Técnicas de Data Augmentation en la Clasificación de Vehículos', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha='right')
    ax.legend(loc='lower right')
    
    # Add data labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(precision_bars)
    add_labels(robustness_bars)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Highlight best approach
    ax.axvspan(x[-1] - width, x[-1] + width, alpha=0.1, color='green')
    
    # Add reference lines
    ax.axhline(y=precision[0], color='royalblue', linestyle='--', alpha=0.5)
    ax.axhline(y=robustness[0], color='firebrick', linestyle='--', alpha=0.5)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01,
               "Robustez medida como precisión en imágenes con perturbaciones controladas (variaciones de iluminación, oclusión y perspectiva).\n"
               "La combinación de técnicas proporciona los mejores resultados tanto en precisión como en robustez.",
               ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('data_augmentation_impact.jpg', dpi=300, bbox_inches='tight')
    print("Data augmentation impact chart saved to 'data_augmentation_impact.jpg'")

def main():
    # Create both visualization plots
    plot_augmentation_examples()
    plot_augmentation_impact()

if __name__ == "__main__":
    main() 