import { writeFile } from 'fs/promises';
import { join } from 'path';
import { execSync } from 'child_process';
import { NextResponse } from 'next/server';

export async function POST(request) {
  try {
    const formData = await request.formData();
    const image = formData.get('image');
    const modelType = formData.get('modelType'); // 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    
    if (!image) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 });
    }
    
    if (!modelType) {
      return NextResponse.json({ error: 'No model type provided' }, { status: 400 });
    }

    // Crear directorios temporales si no existen
    const tempDir = join(process.cwd(), 'temp');
    const inputDir = join(tempDir, 'input');
    const outputDir = join(tempDir, 'output');
    
    try {
      execSync(`mkdir -p ${tempDir} ${inputDir} ${outputDir}`);
    } catch (error) {
      console.error('Error creating directories:', error);
      // En Windows, puede ser que necesitemos usar otro comando
      try {
        execSync(`mkdir "${tempDir}" "${inputDir}" "${outputDir}"`, { shell: 'cmd.exe' });
      } catch (winError) {
        console.error('Error creating directories with Windows command:', winError);
      }
    }

    // Guardar la imagen en el directorio de entrada
    const bytes = await image.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const imagePath = join(inputDir, 'input_image.jpg');
    await writeFile(imagePath, buffer);
    
    // Determinar la ruta a los scripts del servidor
    const serverDir = join(process.cwd(), 'src', 'server');
    const pythonScriptPath = join(serverDir, 'test_resnet.py');
    const cropScriptPath = join(serverDir, 'crop_and_resize.py');
    
    console.log('Rutas a los scripts:');
    console.log('- Server directory:', serverDir);
    console.log('- Python script:', pythonScriptPath);
    console.log('- Crop script:', cropScriptPath);
    
    // Primero, ejecutar el script de crop and resize
    try {
      console.log('Ejecutando script de recorte y redimensionamiento...');
      const cropCommand = `python "${cropScriptPath}" --input "${inputDir}" --output "${outputDir}"`;
      console.log('Comando:', cropCommand);
      const cropResult = execSync(cropCommand, { encoding: 'utf8' });
      console.log('Resultado del crop_and_resize:', cropResult);
    } catch (error) {
      console.error('Error ejecutando script de crop_and_resize:', error.message);
      return NextResponse.json({ 
        error: 'Error al procesar la imagen',
        details: error.message
      }, { status: 500 });
    }

    // Ahora, ejecutar la clasificación con ResNet
    try {
      console.log('Ejecutando script de clasificación...');
      const modelCommand = `python "${pythonScriptPath}" --model ${modelType} --image "${outputDir}/input_image.jpg_crop_car.jpg"`;
      console.log('Comando:', modelCommand);
      const modelResult = execSync(modelCommand, { encoding: 'utf8' });
      console.log('Resultado de la clasificación:', modelResult);
      
      // Intentar parsear el resultado como JSON
      let predictions = [];
      try {
        // Si el script devuelve JSON, lo parseamos
        predictions = JSON.parse(modelResult);
      } catch (parseError) {
        // Si no es JSON válido, intentamos extraer información del texto
        console.error('Error parseando resultado como JSON:', parseError);
        
        // Dividir por líneas y extraer predicciones
        const lines = modelResult.split('\\n');
        for (const line of lines) {
          if (line.includes(':')) {
            const [model, confidence] = line.split(':').map(s => s.trim());
            if (model && confidence) {
              predictions.push({
                model,
                confidence: parseFloat(confidence) || 0
              });
            }
          }
        }
      }
      
      // Si no hay predicciones, usamos un mensaje de error
      if (predictions.length === 0) {
        predictions = [{
          model: 'No se pudieron extraer predicciones',
          confidence: 0,
          rawOutput: modelResult
        }];
      }
      
      // Ordenar predicciones por confianza (de mayor a menor)
      predictions.sort((a, b) => b.confidence - a.confidence);
      
      return NextResponse.json({ 
        predictions,
        modelUsed: modelType,
        processingTime: '1.5', // Valor de ejemplo, podría calcularse con timestamps reales
        rawOutput: modelResult
      });
      
    } catch (error) {
      console.error('Error ejecutando script de clasificación:', error.message);
      return NextResponse.json({ 
        error: 'Error al ejecutar la clasificación',
        details: error.message,
        command: `python "${pythonScriptPath}" --model ${modelType} --image "${outputDir}/input_image.jpg_crop_car.jpg"`
      }, { status: 500 });
    }
    
  } catch (error) {
    console.error('Error general:', error);
    return NextResponse.json({ 
      error: 'Error procesando la solicitud',
      details: error.message
    }, { status: 500 });
  }
} 