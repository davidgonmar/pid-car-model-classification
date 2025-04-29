import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

def create_yolo_resnet_diagram():
    """Create a schematic diagram showing how YOLO and ResNet are used together"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    # Define colors
    colors = {
        'input': '#3498db',     # blue
        'yolo': '#e74c3c',      # red
        'resnet': '#2ecc71',    # green
        'output': '#9b59b6',    # purple
        'data': '#f39c12',      # orange
        'arrow': '#7f8c8d'      # gray
    }
    
    # Define boxes for different components
    components = [
        # (name, x, y, width, height, color, label)
        ('input', 1, 5, 2, 1, colors['input'], 'Input Image'),
        ('yolo', 5, 5, 3, 1, colors['yolo'], 'YOLOv8\nVehicle Detection'),
        ('crop', 10, 5, 2, 1, colors['data'], 'Crop Vehicle ROI'),
        ('resnet', 5, 2, 3, 1, colors['resnet'], 'ResNet-50\nVehicle Classification'),
        ('output', 10, 2, 2, 1, colors['output'], 'Classification Result')
    ]
    
    # Draw components
    for name, x, y, width, height, color, label in components:
        rect = Rectangle((x, y), width, height, facecolor=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Define arrows
    arrows = [
        # (start_component, end_component, label)
        ('input', 'yolo', 'Raw Image'),
        ('yolo', 'crop', 'Bounding Box\nCoordinates'),
        ('crop', 'resnet', 'Cropped\nVehicle Image'),
        ('resnet', 'output', 'Vehicle\nClass')
    ]
    
    # Map components to their coordinates
    comp_coords = {name: (x + width/2, y + height/2) for name, x, y, width, height, _, _ in components}
    
    # Add arrows
    for start, end, label in arrows:
        start_x, start_y = comp_coords[start]
        end_x, end_y = comp_coords[end]
        
        # Create curved arrows
        if start == 'crop' and end == 'resnet':
            # Create a path from crop to resnet (going down)
            arrow = FancyArrowPatch(
                (start_x, start_y - 0.5),  # Start at bottom of crop
                (end_x + 1.5, end_y),      # End at right of resnet
                connectionstyle='arc3,rad=-0.3',
                arrowstyle='-|>', color=colors['arrow'], linewidth=2
            )
            ax.add_patch(arrow)
            ax.text((start_x + end_x + 1.5)/2, (start_y - 0.5 + end_y)/2 - 0.3, label, 
                    ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        else:
            # Create straight arrows for other connections
            arrow = FancyArrowPatch(
                (start_x + 1, start_y) if start == 'input' else 
                (start_x + 1.5, start_y) if start == 'yolo' else
                (start_x + 1, start_y),
                (end_x - 1.5, end_y) if end == 'yolo' else 
                (end_x - 1, end_y) if end == 'crop' else
                (end_x - 1, end_y),
                arrowstyle='-|>', color=colors['arrow'], linewidth=2
            )
            ax.add_patch(arrow)
            
            # Add label to arrow
            label_x = (start_x + end_x) / 2
            label_y = (start_y + end_y) / 2
            ax.text(label_x, label_y, label, ha='center', va='center', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add explanatory text boxes
    explanations = [
        (3, 7, """YOLOv8 Architecture:
- Single-stage detector
- Real-time performance
- Detects vehicle location
- Provides confidence scores""", colors['yolo']),
        
        (3, 0.5, """ResNet-50 Architecture:
- 50-layer deep network
- Skip connections
- Fine-tuned on car dataset
- Classifies vehicle make/model""", colors['resnet']),
        
        (12, 3.5, """Complementary Roles:
1. YOLO identifies vehicle locations
2. ResNet classifies vehicle types
3. Combined pipeline provides
   accurate vehicle identification""", 'lightgray')
    ]
    
    for x, y, text, color in explanations:
        ax.text(x, y, text, fontsize=9, va='center', 
                bbox=dict(facecolor=color, alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Set limits and remove axes
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Set title
    plt.suptitle('YOLO and ResNet Integration for Vehicle Detection & Classification', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.figtext(0.5, 0.01,
                "The system uses YOLOv8 for vehicle detection and ResNet-50 for vehicle classification in a sequential pipeline.\n"
                "This approach combines the strengths of both models: YOLO's ability to quickly locate vehicles and ResNet's accuracy in classification.",
                ha='center', fontsize=10, style='italic')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('yolo_resnet_comparison.jpg', dpi=300, bbox_inches='tight')
    print("YOLO-ResNet integration diagram saved to 'yolo_resnet_comparison.jpg'")

if __name__ == "__main__":
    create_yolo_resnet_diagram() 