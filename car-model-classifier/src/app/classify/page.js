"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { CameraIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { ChevronUpDownIcon } from '@heroicons/react/20/solid';
import Link from 'next/link';
import { cropAndResize } from '@/utils/cropAndResize';
import { RESNET_MODELS } from '@/utils/models';

// Modelo predeterminado (ResNet-50)
const DEFAULT_MODEL = RESNET_MODELS[2];

// Extraer el componente ModelSelector para evitar renderizados innecesarios
function ModelSelector({ selectedModelId, onModelChange, onAnalyze, onReset, canAnalyze, canReset, isProcessing }) {
  return (
    <div className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm">
      <h2 className="text-xl font-semibold mb-4">Select Model</h2>
      
      <div className="relative mt-1">
        <select 
          value={selectedModelId}
          onChange={(e) => onModelChange(e.target.value)}
          className="block w-full appearance-none rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75"
        >
          {RESNET_MODELS.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} - {model.description}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
          <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-blue-50 rounded-md">
        <h3 className="text-sm font-medium text-blue-900 mb-1">Modelo Seleccionado:</h3>
        {(() => {
          const model = RESNET_MODELS.find(m => m.id === selectedModelId);
          return model ? (
            <div>
              <p className="text-blue-700 font-medium">{model.name}</p>
              <p className="text-sm text-blue-600">{model.description}</p>
              <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-blue-700">
                <div>
                  <span className="font-medium">Capas:</span> {model.layers}
                </div>
                <div>
                  <span className="font-medium">Parámetros:</span> {model.parameters}
                </div>
                <div>
                  <span className="font-medium">Precisión:</span> {model.accuracy}
                </div>
                <div>
                  <span className="font-medium">Velocidad:</span> {model.speed}
                </div>
              </div>
            </div>
          ) : null;
        })()}
      </div>
      
      <div className="mt-6 flex gap-4">
        <button
          onClick={onAnalyze}
          disabled={!canAnalyze || isProcessing}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium text-white ${
            !canAnalyze || isProcessing
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isProcessing ? (
            <ArrowPathIcon className="h-5 w-5 animate-spin" />
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
            </svg>
          )}
          {isProcessing ? 'Processing...' : 'Analyze Image'}
        </button>
        
        <button
          onClick={onReset}
          disabled={!canReset || isProcessing}
          className={`px-4 py-2 rounded-lg font-medium  ${
            !canReset || isProcessing
              ? 'text-gray-400 border-gray-300 cursor-not-allowed'
              : 'text-red-600 border-red-200 hover:bg-red-50'
          } border`}
        >
          Reset
        </button>
      </div>
    </div>
  );
}

export default function Classify() {
  // Prevent re-renders with useRef for mutable values that don't need re-render
  const initialMount = useRef(true);
  
  // Almacenar solo el ID del modelo en estado, no todo el objeto
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL.id);
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Obtener el modelo actual basado en el ID
  const selectedModel = RESNET_MODELS.find(model => model.id === selectedModelId) || DEFAULT_MODEL;

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles?.length > 0) {
      const file = acceptedFiles[0];
      
      // Show original image
      const objectUrl = URL.createObjectURL(file);
      setOriginalImage(objectUrl);
      
      // Reset any previous results
      setProcessedImage(null);
      setResults(null);
      setError(null);
      
      try {
        // Process the image (crop and resize)
        setIsProcessing(true);
        const result = await cropAndResize(file);
        setProcessedImage(result.dataUrl);
        
      } catch (err) {
        setError('Error processing image: ' + err.message);
        console.error('Error processing image:', err);
      } finally {
        setIsProcessing(false);
      }
    }
  }, []);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    maxFiles: 1,
    multiple: false
  });

  const handleAnalyze = useCallback(async () => {
    if (!processedImage) return;
    
    setIsProcessing(true);
    setResults(null);
    setError(null);
    
    try {
      // Convert the data URL back to a blob for sending to the API
      const response = await fetch(processedImage);
      const blob = await response.blob();
      
      // Create form data for the API request
      const formData = new FormData();
      formData.append('image', blob, 'processed-image.jpg');
      formData.append('modelType', selectedModelId);
      
      // Call the API endpoint
      const apiResponse = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        throw new Error(`API request failed with status ${apiResponse.status}`);
      }
      
      const data = await apiResponse.json();
      setResults(data);
      
    } catch (err) {
      setError('Error analyzing image: ' + err.message);
      console.error('Error analyzing image:', err);
    } finally {
      setIsProcessing(false);
    }
  }, [processedImage, selectedModelId]);

  const resetAll = useCallback(() => {
    if (originalImage) {
      URL.revokeObjectURL(originalImage);
    }
    setOriginalImage(null);
    setProcessedImage(null);
    setResults(null);
    setError(null);
  }, [originalImage]);

  // Clean up URL objects when component unmounts
  useEffect(() => {
    return () => {
      if (originalImage) {
        URL.revokeObjectURL(originalImage);
      }
    };
  }, [originalImage]);

  // Actualizar la función handleModelChange en el componente principal
  const handleModelChange = useCallback((modelId) => {
    setSelectedModelId(modelId);
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <Link href="/" className="text-blue-600 hover:text-blue-800 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-1">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3" />
          </svg>
          Back to Home
        </Link>
      </div>
      
      <h1 className="text-3xl font-bold mb-8 text-center">Car Model Classification</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left side: Upload and Processing */}
        <div className="space-y-6">
          {/* Image Upload Section */}
          <div className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
            
            <div 
              {...getRootProps()} 
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
              }`}
            >
              <input {...getInputProps()} />
              
              <div className="flex flex-col items-center justify-center space-y-2">
                <CameraIcon className="h-12 w-12 text-gray-400" />
                <p className="text-gray-600">
                  {isDragActive
                    ? "Drop the image here..."
                    : "Drag & drop a car image here, or click to select"}
                </p>
                <p className="text-sm text-gray-500">
                  Supports JPG, JPEG, PNG
                </p>
              </div>
            </div>
            
            {originalImage && (
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">Original Image</h3>
                <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
                  <img 
                    src={originalImage} 
                    alt="Original car" 
                    className="object-contain w-full h-full"
                  />
                </div>
              </div>
            )}
          </div>
          
          {/* Model Selection Section */}
          <ModelSelector 
            selectedModelId={selectedModelId} 
            onModelChange={handleModelChange} 
            onAnalyze={handleAnalyze}
            onReset={resetAll}
            canAnalyze={!!processedImage}
            canReset={!!originalImage}
            isProcessing={isProcessing}
          />
        </div>
        
        {/* Right side: Results */}
        <div className="space-y-6">
          {/* Processed Image */}
          {processedImage && (
            <div className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Processed Image</h2>
              <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                <img 
                  src={processedImage} 
                  alt="Processed car" 
                  className="object-contain w-full h-full"
                />
              </div>
              <p className="mt-2 text-sm text-gray-500 text-center">
                Image cropped and resized to 224×224 for model input
              </p>
            </div>
          )}
          
          {/* Classification Results */}
          {results && (
            <div className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Classification Results</h2>
              
              <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <div className="flex justify-between items-center">
                  <h3 className="font-medium text-blue-900">Model Used</h3>
                  <span className="text-blue-700 font-medium">{results.modelUsed}</span>
                </div>
                <div className="flex justify-between items-center mt-2">
                  <h3 className="font-medium text-blue-900">Processing Time</h3>
                  <span className="text-blue-700 font-medium">{results.processingTime}s</span>
                </div>
              </div>
              
              <h3 className="font-medium text-gray-700 mb-2">Top Predictions</h3>
              <div className="space-y-2">
                {results.predictions && results.predictions.slice(0, 5).map((prediction, index) => (
                  <div 
                    key={index}
                    className={`p-3 rounded-lg ${index === 0 ? 'bg-green-50 border border-green-100' : 'bg-gray-50 border border-gray-100'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className={`font-medium ${index === 0 ? 'text-green-800' : 'text-gray-700'}`}>
                        {prediction.model}
                      </span>
                      <span className={`font-medium ${index === 0 ? 'text-green-800' : 'text-gray-700'}`}>
                        {(prediction.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                      <div 
                        className={`h-2.5 rounded-full ${index === 0 ? 'bg-green-600' : 'bg-blue-600'}`}
                        style={{ width: `${prediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Error message */}
          {error && (
            <div className="border border-red-300 rounded-lg p-6 bg-red-50 text-red-700">
              <h3 className="font-semibold mb-2">Error</h3>
              <p>{error}</p>
            </div>
          )}
          
          {/* Placeholder when no results */}
          {!processedImage && !error && !results && (
            <div className="border border-gray-300 rounded-lg p-12 bg-white shadow-sm flex flex-col items-center justify-center h-full">
              <div className="bg-gray-100 p-6 rounded-full mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-12 h-12 text-gray-400">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-700 mb-2">Results will appear here</h3>
              <p className="text-gray-500 text-center max-w-md">
                Upload an image and click the Analyze button to see the classification results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 