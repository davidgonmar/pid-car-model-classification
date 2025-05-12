#!/usr/bin/env python
import argparse
import sys
import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Añadir el directorio raíz del proyecto al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from lib.resnet import get_model, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    from lib.experiment import get_config
except ImportError:
    print("Error: No se pudieron importar los módulos del proyecto. Verificar rutas.")
    sys.exit(1)

def get_car_predictions(model_type, image_path, top_k=5):
    """
    Realiza predicciones sobre una imagen de carro usando un modelo ResNet.
    
    Args:
        model_type (str): Tipo de modelo ResNet ('resnet18', 'resnet34', etc.)
        image_path (str): Ruta a la imagen
        top_k (int): Número de predicciones top a retornar
        
    Returns:
        str: JSON con lista de predicciones (model, confidence)
    """
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        return json.dumps([{
            "model": "Error",
            "confidence": 0,
            "error": f"Imagen no encontrada: {image_path}"
        }])
    
    try:
        # Cargar modelo
        if model_type == 'resnet18':
            model_class = ResNet18
        elif model_type == 'resnet34':
            model_class = ResNet34
        elif model_type == 'resnet50':
            model_class = ResNet50
        elif model_type == 'resnet101':
            model_class = ResNet101
        elif model_type == 'resnet152':
            model_class = ResNet152
        else:
            model_class = ResNet50
        
        # Intentar cargar el modelo
        try:
            # Intentar cargar configuración de experimento
            config = get_config("1")
            num_classes = 10  # O el número correcto de clases en tu modelo
            
            model = model_class(num_classes=num_classes)
            
            # Intentar cargar pesos preentrenados
            checkpoint_path = os.path.join(project_root, "checkpoints", f"exp_{model_type}.pth")
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                print(f"Modelo cargado desde {checkpoint_path}")
            else:
                # Si no hay checkpoint, usar pesos preentrenados
                model = model_class.from_pretrained(num_classes=num_classes)
                print(f"Utilizando modelo preentrenado {model_type}")
            
            model.eval()
            
            # Preparar transformación de imagen
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            # Cargar y transformar la imagen
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)  # Añadir dimensión de batch
            
            # Hacer predicción
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                
            # Lista de clases (reemplaza con tus clases reales)
            # Normalmente se cargarían de un archivo
            classes = [
                "Toyota Camry", "Honda Civic", "Ford Mustang", "BMW 3 Series",
                "Mercedes-Benz C-Class", "Audi A4", "Chevrolet Corvette",
                "Tesla Model 3", "Porsche 911", "Volkswagen Golf"
            ]
            
            # Obtener top-k predicciones
            top_probs, top_idxs = torch.topk(probabilities, min(top_k, len(classes)))
            
            # Convertir a lista de diccionarios para JSON
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
                predictions.append({
                    "model": classes[idx.item()],
                    "confidence": float(prob.item())
                })
            
            return json.dumps(predictions)
            
        except Exception as e:
            print(f"Error al cargar o ejecutar el modelo: {str(e)}")
            return json.dumps([{
                "model": "Error",
                "confidence": 0,
                "error": f"Error con el modelo: {str(e)}"
            }])
        
    except Exception as e:
        print(f"Error general: {str(e)}")
        return json.dumps([{
            "model": "Error",
            "confidence": 0,
            "error": f"Error general: {str(e)}"
        }])

def main():
    parser = argparse.ArgumentParser(description='Clasificar un modelo de coche con ResNet')
    parser.add_argument('--model', type=str, default='resnet50', help='Modelo a usar (resnet18, resnet34, resnet50, resnet101, resnet152)')
    parser.add_argument('--image', type=str, required=True, help='Ruta a la imagen a clasificar')
    parser.add_argument('--top-k', type=int, default=5, help='Número de predicciones top a mostrar')
    
    args = parser.parse_args()
    
    result = get_car_predictions(args.model, args.image, args.top_k)
    print(result)

if __name__ == "__main__":
    main()
