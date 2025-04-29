import os
import subprocess
import time
import sys


def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["matplotlib", "numpy", "seaborn", "scipy"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Faltan las siguientes librerías: {', '.join(missing_packages)}")
        install = input("¿Desea instalarlas ahora? (s/n): ")
        if install.lower() == "s":
            for package in missing_packages:
                subprocess.check_call(["pip", "install", package])
            print("Librerías instaladas correctamente.")
        else:
            print(
                "Algunas visualizaciones pueden no funcionar sin las librerías necesarias."
            )
    else:
        print("Todas las librerías necesarias están instaladas.")



def run_visualization_scripts():
    """Run all visualization scripts in sequence"""
    scripts = [
        "model_architecture.py",
        "model_comparison.py",
        "gradcam_visualization.py",
        "data_augmentation.py",
        "confusion_matrix.py",
    ]

    print("\n=== Generando todas las visualizaciones del paper ===\n")

    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)

        if os.path.exists(script_path):
            print(f"Ejecutando {script}...")
            try:
                start_time = time.time()
                exec(open(script_path).read())
                end_time = time.time()
                print(f"✓ Completado en {end_time - start_time:.2f} segundos\n")
            except Exception as e:
                print(f"✗ Error al ejecutar {script}: {str(e)}\n")
        else:
            print(f"✗ No se encontró el script {script}\n")

    # List generated files
    print("\n=== Archivos generados ===")
    image_files = [
        f for f in os.listdir(os.path.dirname(__file__)) if f.endswith(".jpg")
    ]

    if image_files:
        for img_file in sorted(image_files):
            file_path = os.path.join(os.path.dirname(__file__), img_file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"{img_file} ({file_size:.1f} KB)")
    else:
        print("No se encontraron archivos de imagen generados.")


def main():
    """Run all plot generation scripts"""
    print("Starting visualization generation process")
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of scripts to run
    scripts = [
        "model_architecture.py",
        "model_comparison.py", 
        "gradcam_visualization.py",
        "data_augmentation.py",
        "confusion_matrix.py",
        "yolo_resnet_comparison.py",
        "tech_dependencies.py",
        "data_augmentation_table.py",
        "slide26_conclusiones.py",
        "slide28_mejoras.py",
        "slide30_final.py"
    ]
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run each script
    successful = 0
    for script in scripts:
        if run_script(script):
            successful += 1
    
    # Report results
    print(f"\nGeneration complete: {successful}/{len(scripts)} visualizations created successfully")
    if successful == len(scripts):
        print("All visualizations were generated successfully!")
    else:
        print(f"Warning: {len(scripts) - successful} script(s) had errors")


if __name__ == "__main__":
    main()
