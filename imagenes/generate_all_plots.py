import os
import subprocess
import time
import sys

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'matplotlib',
        'numpy',
        'seaborn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Faltan las siguientes librerías: {', '.join(missing_packages)}")
        install = input("¿Desea instalarlas ahora? (s/n): ")
        if install.lower() == 's':
            for package in missing_packages:
                subprocess.check_call(['pip', 'install', package])
            print("Librerías instaladas correctamente.")
        else:
            print("Algunas visualizaciones pueden no funcionar sin las librerías necesarias.")
    else:
        print("Todas las librerías necesarias están instaladas.")

def run_script(script_name):
    """Run a Python script and report any errors"""
    print(f"\nRunning {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], 
                               check=True, 
                               capture_output=True, 
                               text=True)
        print(f"✓ {script_name} completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script_name}:")
        print(e.stderr)
        return False

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