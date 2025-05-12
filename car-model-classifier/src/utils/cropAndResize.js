// Simplified version of the crop_and_resize.py functionality for browser
// This is a simplified version as the original uses YOLO and OpenCV which are not easily transferable to a web app

export async function cropAndResize(imageFile) {
  return new Promise((resolve, reject) => {
    try {
      const img = new Image();
      const url = URL.createObjectURL(imageFile);
      
      img.onload = () => {
        // Create a canvas element to manipulate the image
        const canvas = document.createElement('canvas');
        
        // Set dimensions to 224x224 as required by ResNet models
        const targetSize = 224;
        canvas.width = targetSize;
        canvas.height = targetSize;
        
        const ctx = canvas.getContext('2d');
        
        // Simple resize logic (in a real app with YOLO, we would detect and crop the car first)
        // For now we'll just resize and maintain aspect ratio
        let sourceX = 0;
        let sourceY = 0;
        let sourceWidth = img.width;
        let sourceHeight = img.height;
        
        // Calculate dimensions to maintain aspect ratio
        if (img.width > img.height) {
          sourceWidth = img.height;
          sourceX = (img.width - img.height) / 2;
        } else {
          sourceHeight = img.width;
          sourceY = (img.height - img.width) / 2;
        }
        
        // Draw the image on the canvas with the calculated dimensions
        ctx.drawImage(
          img,
          sourceX, sourceY, sourceWidth, sourceHeight,
          0, 0, targetSize, targetSize
        );
        
        // Convert the canvas to a blob
        canvas.toBlob((blob) => {
          resolve({
            processedImage: blob,
            dataUrl: canvas.toDataURL('image/jpeg')
          });
          
          // Clean up
          URL.revokeObjectURL(url);
        }, 'image/jpeg', 0.95);
      };
      
      img.onerror = () => {
        reject(new Error('Failed to load image'));
        URL.revokeObjectURL(url);
      };
      
      img.src = url;
    } catch (error) {
      reject(error);
    }
  });
} 