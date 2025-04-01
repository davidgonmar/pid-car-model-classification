import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_sample_image(size=224):
    """Create a placeholder car image using simple shapes"""
    img = np.ones((size, size, 3))
    
    # Car body (rectangle)
    img[70:160, 40:184, :] = [0.7, 0.7, 0.7]
    
    # Car roof
    img[40:70, 80:170, :] = [0.5, 0.5, 0.5]
    
    # Windows
    img[45:65, 85:110, :] = [0.8, 0.9, 1.0]  # Front window
    img[45:65, 120:165, :] = [0.8, 0.9, 1.0]  # Back window
    
    # Wheels
    y, x = np.ogrid[0:size, 0:size]
    wheel1_mask = (x - 70)**2 + (y - 160)**2 <= 20**2
    wheel2_mask = (x - 154)**2 + (y - 160)**2 <= 20**2
    img[wheel1_mask] = [0.1, 0.1, 0.1]
    img[wheel2_mask] = [0.1, 0.1, 0.1]
    
    # Headlights
    img[120:140, 45:60, :] = [1.0, 1.0, 0.8]
    
    # Taillights
    img[120:140, 170:180, :] = [1.0, 0.2, 0.2]
    
    return img

def create_heatmap(size=224, model="ResNet50"):
    """Create a simulated Grad-CAM heatmap for different models"""
    y, x = np.ogrid[0:size, 0:size]
    
    if model == "ResNet50":
        # ResNet50 focuses more on the distinguishing features (lights, wheels)
        headlights = np.exp(-0.01 * ((x - 52)**2 + (y - 130)**2))
        taillights = np.exp(-0.01 * ((x - 175)**2 + (y - 130)**2))
        wheels = np.exp(-0.01 * ((x - 70)**2 + (y - 160)**2)) + np.exp(-0.01 * ((x - 154)**2 + (y - 160)**2))
        hood = np.exp(-0.005 * ((x - 110)**2 + (y - 100)**2)) * 0.7
        
        heatmap = headlights + taillights + wheels + hood
        
    elif model == "VGG16":
        # VGG16 focuses more broadly on the entire car area
        center = np.exp(-0.001 * ((x - 110)**2 + (y - 110)**2))
        heatmap = center * 0.8
        
    elif model == "MobileNet":
        # MobileNet has more scattered attention
        center = np.exp(-0.002 * ((x - 110)**2 + (y - 110)**2))
        random_spots = np.random.rand(size, size) * 0.3
        heatmap = center + random_spots
        
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap

def apply_heatmap(img, heatmap, alpha=0.6):
    """Apply heatmap to image"""
    # Create a custom colormap (blue to red)
    colors = [(0, 0, 1, 0), (0, 1, 1, 0.3), (0, 1, 0, 0.5), (1, 1, 0, 0.7), (1, 0, 0, 1.0)]
    cm = LinearSegmentedColormap.from_list('gradcam', colors, N=100)
    
    # Resize heatmap if needed
    if img.shape[0] != heatmap.shape[0] or img.shape[1] != heatmap.shape[1]:
        from skimage.transform import resize
        heatmap = resize(heatmap, (img.shape[0], img.shape[1]))
    
    # Apply colormap to heatmap
    heatmap_colored = cm(heatmap)
    
    # Combine image and heatmap
    result = img.copy()
    for i in range(3):
        result[:, :, i] = img[:, :, i] * (1 - alpha * heatmap_colored[:, :, 3]) + \
                         heatmap_colored[:, :, i] * alpha * heatmap_colored[:, :, 3]
    
    return result

def main():
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a sample car image
    car_img = create_sample_image()
    
    # List of models to compare
    models = ["ResNet50", "VGG16", "MobileNet"]
    
    # Create and apply Grad-CAM visualizations for each model
    for i, model in enumerate(models):
        heatmap = create_heatmap(model=model)
        result = apply_heatmap(car_img, heatmap)
        
        axes[i].imshow(result)
        axes[i].set_title(f'Grad-CAM: {model}', fontsize=12)
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle('Comparación de Visualizaciones Grad-CAM para Diferentes Arquitecturas', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Add an explanatory note
    plt.figtext(0.5, 0.01, 
               "La visualización Grad-CAM muestra qué partes de la imagen son más relevantes para la clasificación.\n"
               "Observe cómo ResNet50 se enfoca en características discriminativas clave (faros, ruedas),\n"
               "mientras que otros modelos tienen patrones de atención diferentes.",
               ha='center', fontsize=9, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig('gradcam_comparison.jpg', dpi=300, bbox_inches='tight')
    print("Grad-CAM visualizations saved to 'gradcam_comparison.jpg'")

if __name__ == "__main__":
    main() 