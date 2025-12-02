#import numpy as np
#import matplotlib
#matplotlib.use('Agg')  # Fix for WSL/headless environment
#import matplotlib.pyplot as plt
#import cv2
#from keras.models import Model, load_model
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input
#import tensorflow as tf
#
#class GradCAM:
#    def __init__(self, model, layer_name):
#        """
#        Initialize Grad-CAM with your trained model
#        
#        Args:
#            model: Your trained Keras model
#            layer_name: Name of the convolutional layer to visualize (e.g., 'conv2d_2' for the last conv layer)
#        """
#        self.model = model
#        self.layer_name = layer_name
#        
#        # Create a model that outputs both predictions and the target layer's output
#        self.grad_model = Model(
#            inputs=[self.model.inputs],
#            outputs=[self.model.get_layer(layer_name).output, self.model.output]
#        )
#    
#    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
#        """
#        Compute Grad-CAM heatmap
#        
#        Args:
#            img_array: Preprocessed image array (1, 100, 100, 3)
#            class_idx: Target class index (if None, uses predicted class)
#            eps: Small value to avoid division by zero
#        
#        Returns:
#            heatmap: Grad-CAM heatmap as numpy array
#        """
#        # Record operations for automatic differentiation
#        with tf.GradientTape() as tape:
#            # Get the convolutional output and predictions
#            conv_outputs, predictions = self.grad_model(img_array)
#            
#            # If class_idx is None, use the predicted class
#            if class_idx is None:
#                class_idx = tf.argmax(predictions[0])
#            
#            # Get the score for the target class
#            class_channel = predictions[:, class_idx]
#        
#        # Compute gradients of the class score with respect to the feature map
#        grads = tape.gradient(class_channel, conv_outputs)
#        
#        # Global average pooling of gradients
#        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#        
#        # Weight the feature maps by the gradients
#        conv_outputs = conv_outputs[0]
#        pooled_grads = pooled_grads.numpy()
#        conv_outputs = conv_outputs.numpy()
#        
#        # Multiply each feature map by its gradient weight
#        for i in range(pooled_grads.shape[-1]):
#            conv_outputs[:, :, i] *= pooled_grads[i]
#        
#        # Average over all feature maps to get the heatmap
#        heatmap = np.mean(conv_outputs, axis=-1)
#        
#        # Normalize the heatmap
#        heatmap = np.maximum(heatmap, 0)  # ReLU
#        heatmap /= (np.max(heatmap) + eps)  # Normalize to [0, 1]
#        
#        return heatmap
#    
#    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
#        """
#        Overlay heatmap on original image with smooth interpolation
#        
#        Args:
#            heatmap: Grad-CAM heatmap
#            original_img: Original image (as numpy array)
#            alpha: Transparency of overlay (0-1)
#            colormap: OpenCV colormap to use
#        
#        Returns:
#            superimposed_img: Image with heatmap overlay
#        """
#        # Resize heatmap with cubic interpolation for smoother results
#        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), 
#                                     interpolation=cv2.INTER_CUBIC)
#        
#        # Apply Gaussian blur for smoother appearance
#        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
#        
#        # Normalize again after blurring
#        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
#        
#        # Convert heatmap to RGB
#        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
#        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
#        
#        # Superimpose the heatmap on original image
#        superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
#        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
#        
#        return superimposed_img
#
#
#def visualize_gradcam(model_path, image_path, class_names, save_path='gradcam_result.png'):
#    """
#    Complete pipeline to generate and save Grad-CAM visualization
#    
#    Args:
#        model_path: Path to your trained model (.h5 file)
#        image_path: Path to the image you want to visualize
#        class_names: List of class names ["BKL", "MEL2", "NV", "BCC"]
#        save_path: Where to save the visualization
#    """
#    # Load the trained model
#    model = load_model(model_path)
#    print("Model loaded successfully!")
#    print("\nModel layers:")
#    for i, layer in enumerate(model.layers):
#        print(f"{i}: {layer.name} - {layer.__class__.__name__}")
#    
#    # Load and preprocess the image
#    img = image.load_img(image_path, target_size=(100, 100))
#    img_array = image.img_to_array(img)
#    original_img = img_array.copy()
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array = img_array / 255.0  # Normalize to [0, 1] as per your training
#    
#    # Make prediction
#    predictions = model.predict(img_array)
#    predicted_class_idx = np.argmax(predictions[0])
#    predicted_class = class_names[predicted_class_idx]
#    confidence = predictions[0][predicted_class_idx]
#    
#    print(f"\nPrediction: {predicted_class} (Confidence: {confidence:.2%})")
#    print(f"All probabilities: {dict(zip(class_names, predictions[0]))}")
#    
#    # Initialize Grad-CAM with an earlier convolutional layer for better resolution
#    # conv2d_1 (32 filters) gives smoother results than conv2d_2 (64 filters)
#    # You can also try 'conv2d' for even higher resolution
#    gradcam = GradCAM(model, layer_name='conv2d_1')
#    
#    # Compute heatmap
#    heatmap = gradcam.compute_heatmap(img_array, class_idx=predicted_class_idx)
#    
#    # Create overlay
#    superimposed_img = gradcam.overlay_heatmap(heatmap, original_img)
#    
#    # Create high-quality visualization
#    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#    
#    # Original image
#    axes[0].imshow(original_img.astype(np.uint8))
#    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
#    axes[0].axis('off')
#    
#    # Heatmap only
#    axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
#    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
#    axes[1].axis('off')
#    
#    # Overlay
#    axes[2].imshow(superimposed_img)
#    axes[2].set_title(f'Grad-CAM Overlay\n{predicted_class} ({confidence:.2%})', 
#                     fontsize=14, fontweight='bold')
#    axes[2].axis('off')
#    
#    plt.tight_layout()
#    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
#    print(f"\nVisualization saved to: {save_path}")
#    
#    return heatmap, superimposed_img, predicted_class
#
#
#def compare_layers(model_path, image_path, class_names, layers=['conv2d', 'conv2d_1', 'conv2d_2'], save_path='gradcam_comparison.png'):
#    """
#    Compare Grad-CAM visualizations from different convolutional layers
#    
#    Args:
#        model_path: Path to your trained model
#        image_path: Path to the image
#        class_names: List of class names
#        layers: List of layer names to compare
#        save_path: Where to save the comparison
#    """
#    model = load_model(model_path)
#    
#    # Load and preprocess image
#    img = image.load_img(image_path, target_size=(100, 100))
#    img_array = image.img_to_array(img)
#    original_img = img_array.copy()
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array = img_array / 255.0
#    
#    # Make prediction
#    predictions = model.predict(img_array)
#    predicted_class_idx = np.argmax(predictions[0])
#    predicted_class = class_names[predicted_class_idx]
#    confidence = predictions[0][predicted_class_idx]
#    
#    # Create comparison plot
#    fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers), 6))
#    if len(layers) == 1:
#        axes = [axes]
#    
#    for i, layer_name in enumerate(layers):
#        gradcam = GradCAM(model, layer_name=layer_name)
#        heatmap = gradcam.compute_heatmap(img_array, class_idx=predicted_class_idx)
#        overlay = gradcam.overlay_heatmap(heatmap, original_img)
#        
#        axes[i].imshow(overlay)
#        axes[i].set_title(f'Layer: {layer_name}', fontsize=12, fontweight='bold')
#        axes[i].axis('off')
#    
#    plt.suptitle(f'Prediction: {predicted_class} ({confidence:.2%})', fontsize=14, fontweight='bold')
#    plt.tight_layout()
#    plt.savefig(save_path, dpi=300, bbox_inches='tight')
#    print(f"\nLayer comparison saved to: {save_path}")
#
#
## Example usage:
#if __name__ == "__main__":
#    # Configuration
#    MODEL_PATH = 'best_model5_new.h5'
#    #IMAGE_PATH='/home/nehal/workspace/cancer_classification/SIDE+REAR/BCC/correct/BCC/copy_8_isic_0059159 (1314).jpg'
#    IMAGE_PATH = '/home/nehal/workspace/cancer_classification/train/New folder/val/BKL/ISIC_0156460.jpg'
#    CLASS_NAMES = ["BKL", "MEL", "NV", "BCC"]
#    # Generate Grad-CAM visualization
#    heatmap, overlay, prediction = visualize_gradcam(
#        model_path=MODEL_PATH,
#        image_path=IMAGE_PATH,
#        class_names=CLASS_NAMES,
#        save_path='gradcam_visualization.png'
#    )
#    
#    print("\nDone! Check 'gradcam_visualization.png' for results.")
#    
#    # Optional: Compare different layers
#    print("\nGenerating layer comparison...")
#    compare_layers(
#        model_path=MODEL_PATH,
#        image_path=IMAGE_PATH,
#        class_names=CLASS_NAMES,
#        layers=['conv2d', 'conv2d_1', 'conv2d_2'],
#        save_path='gradcam_layer_comparison.png'
#    )

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix for WSL/headless environment
import matplotlib.pyplot as plt
import cv2
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import os
import glob
from pathlib import Path

class GradCAM:
    def __init__(self, model, layer_name):
        """
        Initialize Grad-CAM with your trained model
        
        Args:
            model: Your trained Keras model
            layer_name: Name of the convolutional layer to visualize (e.g., 'conv2d_2' for the last conv layer)
        """
        self.model = model
        self.layer_name = layer_name
        
        # Create a model that outputs both predictions and the target layer's output
        self.grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
    
    def compute_heatmap(self, img_array, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            img_array: Preprocessed image array (1, 100, 100, 3)
            class_idx: Target class index (if None, uses predicted class)
            eps: Small value to avoid division by zero
        
        Returns:
            heatmap: Grad-CAM heatmap as numpy array
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get the convolutional output and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # If class_idx is None, use the predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of the class score with respect to the feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each feature map by its gradient weight
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all feature maps to get the heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= (np.max(heatmap) + eps)  # Normalize to [0, 1]
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image with smooth interpolation
        
        Args:
            heatmap: Grad-CAM heatmap
            original_img: Original image (as numpy array)
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap to use
        
        Returns:
            superimposed_img: Image with heatmap overlay
        """
        # Resize heatmap with cubic interpolation for smoother results
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), 
                                     interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur for smoother appearance
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (11, 11), 0)
        
        # Normalize again after blurring
        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose the heatmap on original image
        superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img


def visualize_gradcam(model, gradcam, image_path, class_names, save_path='gradcam_result.png'):
    """
    Complete pipeline to generate and save Grad-CAM visualization
    
    Args:
        model: Loaded Keras model
        gradcam: GradCAM object
        image_path: Path to the image you want to visualize
        class_names: List of class names ["BKL", "MEL2", "NV", "BCC"]
        save_path: Where to save the visualization
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    original_img = img_array.copy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1] as per your training
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Compute heatmap
    heatmap = gradcam.compute_heatmap(img_array, class_idx=predicted_class_idx)
    
    # Create overlay
    superimposed_img = gradcam.overlay_heatmap(heatmap, original_img)
    
    # Create high-quality visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img.astype(np.uint8))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet', interpolation='bilinear')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Grad-CAM Overlay\n{predicted_class} ({confidence:.2%})', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Higher DPI for better quality
    plt.close()  # Close to free memory
    
    return heatmap, superimposed_img, predicted_class, confidence


def compare_layers_batch(model, image_path, class_names, layers=['conv2d', 'conv2d_1', 'conv2d_2'], save_path='gradcam_comparison.png'):
    """
    Compare Grad-CAM visualizations from different convolutional layers
    
    Args:
        model: Loaded Keras model
        image_path: Path to the image
        class_names: List of class names
        layers: List of layer names to compare
        save_path: Where to save the comparison
    """
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    original_img = img_array.copy()
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers), 6))
    if len(layers) == 1:
        axes = [axes]
    
    for i, layer_name in enumerate(layers):
        gradcam = GradCAM(model, layer_name=layer_name)
        heatmap = gradcam.compute_heatmap(img_array, class_idx=predicted_class_idx)
        overlay = gradcam.overlay_heatmap(heatmap, original_img)
        
        axes[i].imshow(overlay)
        axes[i].set_title(f'Layer: {layer_name}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Prediction: {predicted_class} ({confidence:.2%})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_directory(model_path, input_dir, output_dir, class_names, layer_name='conv2d_1', compare_layers_list=['conv2d', 'conv2d_1', 'conv2d_2']):
    """
    Process all .jpg images in a directory and save Grad-CAM visualizations
    
    Args:
        model_path: Path to your trained model (.h5 file)
        input_dir: Directory containing .jpg images to process
        output_dir: Directory to save results
        class_names: List of class names
        layer_name: Which conv layer to use for main Grad-CAM
        compare_layers_list: List of layers to compare
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model once
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Initialize Grad-CAM once
    gradcam = GradCAM(model, layer_name=layer_name)
    
    # Find all .jpg files (case-insensitive)
    jpg_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        jpg_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    if not jpg_files:
        print(f"No .jpg files found in {input_dir}")
        return
    
    print(f"\nFound {len(jpg_files)} .jpg files to process")
    print(f"Results will be saved to: {output_dir}\n")
    
    # Process each image
    results = []
    for idx, img_path in enumerate(jpg_files, 1):
        try:
            # Get relative path and create output filename
            rel_path = os.path.relpath(img_path, input_dir)
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Create subdirectory structure in output
            rel_dir = os.path.dirname(rel_path)
            out_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)
            
            # Output paths
            save_path_main = os.path.join(out_subdir, f'{name_without_ext}_gradcam.png')
            save_path_comparison = os.path.join(out_subdir, f'{name_without_ext}_layer_comparison.png')
            
            print(f"[{idx}/{len(jpg_files)}] Processing: {rel_path}")
            
            # Generate main Grad-CAM visualization
            heatmap, overlay, prediction, confidence = visualize_gradcam(
                model=model,
                gradcam=gradcam,
                image_path=img_path,
                class_names=class_names,
                save_path=save_path_main
            )
            
            # Generate layer comparison
            compare_layers_batch(
                model=model,
                image_path=img_path,
                class_names=class_names,
                layers=compare_layers_list,
                save_path=save_path_comparison
            )
            
            print(f"    → Prediction: {prediction} ({confidence:.2%})")
            print(f"    → Saved main visualization to: {save_path_main}")
            print(f"    → Saved layer comparison to: {save_path_comparison}\n")
            
            results.append({
                'image': rel_path,
                'prediction': prediction,
                'confidence': confidence,
                'output_main': save_path_main,
                'output_comparison': save_path_comparison
            })
            
        except Exception as e:
            print(f"    ✗ Error processing {img_path}: {str(e)}\n")
            continue
    
    # Save summary
    summary_path = os.path.join(output_dir, 'processing_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Grad-CAM Processing Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total images processed: {len(results)}/{len(jpg_files)}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Main layer: {layer_name}\n")
        f.write(f"Comparison layers: {', '.join(compare_layers_list)}\n\n")
        f.write("Output files generated per image:\n")
        f.write("  1. {name}_gradcam.png - Main 3-panel visualization\n")
        f.write("  2. {name}_layer_comparison.png - Layer comparison\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Image':<50} {'Prediction':<10} {'Confidence':<12}\n")
        f.write("-" * 80 + "\n")
        
        for r in results:
            f.write(f"{r['image']:<50} {r['prediction']:<10} {r['confidence']:<12.2%}\n")
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Processed: {len(results)}/{len(jpg_files)} images")
    print(f"Generated {len(results) * 2} output files (2 per image)")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}")


# Example usage:
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'best_model5_new.h5'
    INPUT_DIR = '/home/nehal/workspace/cancer_classification/train/NV'
    OUTPUT_DIR = '/home/nehal/workspace/cancer_classification/gradcam_results'
    CLASS_NAMES = ["BKL", "MEL", "NV", "BCC"]
    
    # Process all images in directory
    # This will generate 2 files per image:
    # 1. {name}_gradcam.png - Main 3-panel visualization
    # 2. {name}_layer_comparison.png - Layer comparison across conv layers
    process_directory(
        model_path=MODEL_PATH,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        class_names=CLASS_NAMES,
        layer_name='conv2d_1',  # Main visualization layer
        compare_layers_list=['conv2d', 'conv2d_1', 'conv2d_2']  # Layers to compare
    )
    
    print("\nDone! Check the output directory for all results.")
    print("Each image generates 2 files:")
    print("  - {name}_gradcam.png (3-panel visualization)")
    print("  - {name}_layer_comparison.png (layer comparison)")