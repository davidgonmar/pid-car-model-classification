# Car Model Classification using PyTorch

This project implements a deep learning model to classify different car models using PyTorch. The system is designed to identify and classify various car makes and models from images with high accuracy.

## 🚗 Project Overview

This project uses state-of-the-art deep learning techniques to classify car models from images. It leverages transfer learning with pre-trained models fine-tuned on a custom dataset of car images.

### Key Features

- Deep learning-based car model classification
- Transfer learning using pre-trained models
- Support for multiple car makes and models
- High accuracy prediction capabilities
- Real-time inference support
- High-precision vehicle detection and cropping using YOLOv8

## 🛠️ Technical Stack

- **Python 3.8+**
- **PyTorch**: Main deep learning framework
- **torchvision**: For image processing and pre-trained models
- **PIL**: For image handling
- **NumPy**: For numerical computations
- **Matplotlib**: For visualization
- **YOLOv8**: For precise vehicle detection and cropping
- **OpenCV**: For image processing operations

## 📁 Project Structure

```
pid-car-model-classification/
├── data/                    # Dataset directory
│   ├── raw/                # Original images
│   ├── processed/          # Processed images
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
├── models/                 # Model architecture definitions
├── utils/                  # Utility functions
│   └── crop_and_resize.py  # Vehicle detection and cropping
├── train.py               # Training script
├── predict.py             # Inference script
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/davidgonmar/pid-car-model-classification.git
cd pid-car-model-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Model Architecture

The project uses a convolutional neural network (CNN) based on state-of-the-art architectures like ResNet or EfficientNet, fine-tuned for car model classification. The model is trained to recognize various features that distinguish different car models, including:
- Body shape and design
- Front and rear appearances
- Distinctive brand features
- Model-specific characteristics

## 🎯 Training

The model is trained using:
- Transfer learning from pre-trained models
- Data augmentation techniques
- Learning rate scheduling
- Early stopping
- Model checkpointing

## 📈 Performance

The model aims to achieve:
- High accuracy in car model classification
- Fast inference time
- Robustness to different lighting conditions and angles
- Good generalization to unseen car models

## 🔍 Vehicle Detection and Cropping

The project uses YOLOv8, a state-of-the-art object detection model, to accurately detect and crop vehicles from images. This high-precision cropping is crucial for the subsequent vehicle classification task, as it ensures that:

1. Only the vehicle of interest is included in the cropped image
2. The vehicle is properly centered and framed
3. The background noise is minimized
4. The aspect ratio is maintained for consistent model input

The cropping process (`utils/crop_and_resize.py`) follows these steps:

1. **Vehicle Detection**: Uses YOLOv8 to detect vehicles in the image with high confidence
2. **Bounding Box Extraction**: Obtains precise bounding boxes around detected vehicles
3. **Image Cropping**: Crops the image using the detected bounding boxes while maintaining the original aspect ratio
4. **Background Preservation**: Keeps the original background to maintain context and avoid artificial edges
5. **Quality Control**: Ensures the cropped images maintain high quality for accurate model classification

This approach is essential for:
- Improving classification accuracy by focusing on the relevant vehicle features
- Reducing noise from background elements
- Maintaining consistent input format for the classification model
- Preserving important contextual information

### Usage

```bash
python utils/crop_and_resize.py --input_dir path/to/input/images --output_dir path/to/output/images
```

The script will process all images in the input directory and save the cropped vehicle images to the output directory, maintaining the original image quality and aspect ratio.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For any queries or suggestions, please open an issue in the repository.
