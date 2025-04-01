import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

def create_conv_layer(ax, x, y, width, height, filters, name):
    """Create a convolutional layer visualization"""
    # Main box
    rect = Rectangle((x, y), width, height, facecolor='skyblue', edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    
    # Add text
    ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8)
    ax.text(x + width/2, y + height/4, f"{filters} filters", ha='center', va='center', fontsize=6)

def create_pool_layer(ax, x, y, width, height, name):
    """Create a pooling layer visualization"""
    rect = Rectangle((x, y), width, height, facecolor='lightgreen', edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8)

def create_fc_layer(ax, x, y, width, height, neurons, name):
    """Create a fully connected layer visualization"""
    rect = Rectangle((x, y), width, height, facecolor='salmon', edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8)
    ax.text(x + width/2, y + height/4, f"{neurons} neurons", ha='center', va='center', fontsize=6)

def create_arrow(ax, x1, y1, x2, y2):
    """Create an arrow between layers"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', connectionstyle='arc3,rad=0',
                          mutation_scale=10, linewidth=1)
    ax.add_patch(arrow)

def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the axes
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_axis_off()
    
    # Network title
    ax.text(6, 5.5, 'ResNet50 Architecture for Vehicle Classification', ha='center', fontsize=14, weight='bold')
    
    # Input layer
    input_rect = Rectangle((0.5, 2.5), 1, 1, facecolor='lightgray', edgecolor='black')
    ax.add_patch(input_rect)
    ax.text(1, 3, "Input\n224x224x3", ha='center', va='center', fontsize=8)
    
    # First few layers
    create_conv_layer(ax, 2, 2.5, 1, 1, 64, "Conv1\n7x7")
    create_pool_layer(ax, 3.5, 2.5, 1, 1, "MaxPool\n3x3")
    
    # ResNet blocks (simplified)
    create_conv_layer(ax, 5, 2.5, 1.5, 1, "64-256", "Res Block 1\nx3")
    create_conv_layer(ax, 7, 2.5, 1.5, 1, "128-512", "Res Block 2\nx4")
    create_conv_layer(ax, 9, 2.5, 1.5, 1, "256-1024", "Res Block 3\nx6")
    
    # Final layers
    create_pool_layer(ax, 11, 2.5, 0.5, 1, "AvgPool")
    create_fc_layer(ax, 11, 4, 0.5, 0.8, "N", "FC\nOutput")
    
    # Arrows
    create_arrow(ax, 1.5, 3, 2, 3)
    create_arrow(ax, 3, 3, 3.5, 3)
    create_arrow(ax, 4.5, 3, 5, 3)
    create_arrow(ax, 6.5, 3, 7, 3)
    create_arrow(ax, 8.5, 3, 9, 3)
    create_arrow(ax, 10.5, 3, 11, 3)
    create_arrow(ax, 11.25, 3.5, 11.25, 4)
    
    # Legend
    legend_x = 1
    legend_y = 1.5
    
    # Convolutional
    ax.add_patch(Rectangle((legend_x, legend_y), 0.3, 0.3, facecolor='skyblue', edgecolor='black', alpha=0.8))
    ax.text(legend_x + 0.4, legend_y + 0.15, "Convolutional Layer", fontsize=7, va='center')
    
    # Pooling
    ax.add_patch(Rectangle((legend_x + 3, legend_y), 0.3, 0.3, facecolor='lightgreen', edgecolor='black', alpha=0.8))
    ax.text(legend_x + 3.4, legend_y + 0.15, "Pooling Layer", fontsize=7, va='center')
    
    # Fully Connected
    ax.add_patch(Rectangle((legend_x + 6, legend_y), 0.3, 0.3, facecolor='salmon', edgecolor='black', alpha=0.8))
    ax.text(legend_x + 6.4, legend_y + 0.15, "Fully Connected Layer", fontsize=7, va='center')
    
    # Add additional text
    ax.text(6, 0.7, "Total params: ~25.6M | Depth: 50 layers | Original paper: He et al. (2016)", 
            ha='center', fontsize=8, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('resnet50_architecture.jpg', dpi=300, bbox_inches='tight')
    print("ResNet50 architecture diagram saved to 'resnet50_architecture.jpg'")

if __name__ == "__main__":
    main() 