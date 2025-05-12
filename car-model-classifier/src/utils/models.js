/**
 * Information about the available ResNet models
 */

export const RESNET_MODELS = [
  {
    id: 'resnet18',
    name: 'ResNet-18',
    description: 'Faster, smaller model',
    layers: 18,
    parameters: '11 million',
    accuracy: '~70%',
    speed: 'Very Fast'
  },
  {
    id: 'resnet34',
    name: 'ResNet-34',
    description: 'Good balance of speed and accuracy',
    layers: 34,
    parameters: '21 million',
    accuracy: '~74%',
    speed: 'Fast'
  },
  {
    id: 'resnet50',
    name: 'ResNet-50',
    description: 'High accuracy, moderate size',
    layers: 50,
    parameters: '25 million',
    accuracy: '~76%',
    speed: 'Moderate'
  },
  {
    id: 'resnet101',
    name: 'ResNet-101',
    description: 'Very high accuracy, larger model',
    layers: 101,
    parameters: '44 million',
    accuracy: '~78%',
    speed: 'Slow'
  },
  {
    id: 'resnet152',
    name: 'ResNet-152',
    description: 'Highest accuracy, largest model',
    layers: 152,
    parameters: '60 million',
    accuracy: '~79%',
    speed: 'Very Slow'
  }
];

/**
 * Get a model by its ID
 * @param {string} id - The model ID
 * @returns {object|undefined} The model object or undefined if not found
 */
export function getModelById(id) {
  return RESNET_MODELS.find(model => model.id === id);
}

/**
 * Get detailed information about a model
 * @param {string} id - The model ID 
 * @returns {string} HTML-formatted information about the model
 */
export function getModelInfo(id) {
  const model = getModelById(id);
  if (!model) return '';
  
  return `
    <p><strong>Architecture:</strong> ${model.name}</p>
    <p><strong>Layers:</strong> ${model.layers}</p>
    <p><strong>Parameters:</strong> ${model.parameters}</p>
    <p><strong>Accuracy:</strong> ${model.accuracy} (ImageNet)</p>
    <p><strong>Inference Speed:</strong> ${model.speed}</p>
  `;
} 