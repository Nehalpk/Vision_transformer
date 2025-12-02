import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def estimate_lesion_coverage(image_path):
    """
    Estimates lesion coverage percentage and recommends patch size for dermoscopic images.
    
    Args:
        image_path: Path to the dermoscopic image (JPG)
    
    Returns:
        dict: Contains coverage percentage, patch size, lesion area, and processed images
    """
    
    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Step 1: Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Step 1: Converted to grayscale")
    
    # Step 2: Otsu's Automatic Thresholding
    # Otsu's method automatically calculates optimal threshold
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("Step 2: Applied Otsu's thresholding")
    
    # Step 3: Morphological Operations
    # Define kernels for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Morphological Opening (erosion followed by dilation)
    # Removes hair artifacts and small noise spots
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Morphological Closing (dilation followed by erosion)
    # Fills small holes within lesion boundary
    cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    print("Step 3: Applied morphological operations (opening + closing)")
    
    # Step 4: Connected Component Analysis
    # Find all connected components and select the largest one
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # Ignore background (label 0), find largest component
    if num_labels > 1:
        # Get areas of all components (excluding background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1  # +1 because we excluded background
        
        # Create mask with only the largest component
        final_mask = np.zeros_like(cleaned_mask)
        final_mask[labels == largest_component_label] = 255
        
        lesion_area = areas[largest_component_label - 1]
        print(f"Step 4: Selected largest connected component (area: {lesion_area} pixels)")
    else:
        final_mask = cleaned_mask
        lesion_area = np.sum(cleaned_mask == 255)
        print("Step 4: No connected components found, using full mask")
    
    # Step 5: Coverage Calculation
    total_area = img.shape[0] * img.shape[1]
    coverage_percentage = (lesion_area / total_area) * 100
    print(f"Step 5: Calculated coverage = {coverage_percentage:.2f}%")
    
    # Step 6: Dynamic Patch Size Selection
    if coverage_percentage < 25:
        patch_size = 8
        num_patches = (img.shape[0] // patch_size) * (img.shape[1] // patch_size)
        recommendation = "Small lesions → 8×8 patches (finer granularity for morphological details)"
    elif 25 <= coverage_percentage <= 50:
        patch_size = 16
        num_patches = (img.shape[0] // patch_size) * (img.shape[1] // patch_size)
        recommendation = "Medium lesions → 16×16 patches (balanced resolution and efficiency)"
    else:  # coverage_percentage > 50
        patch_size = 32
        num_patches = (img.shape[0] // patch_size) * (img.shape[1] // patch_size)
        recommendation = "Large lesions → 32×32 patches (maintains features, reduces computation)"
    
    print(f"Step 6: Recommended patch size = {patch_size}×{patch_size} ({num_patches} patches)")
    
    # Return results
    results = {
        'coverage_percentage': coverage_percentage,
        'lesion_area': lesion_area,
        'total_area': total_area,
        'patch_size': patch_size,
        'num_patches': num_patches,
        'recommendation': recommendation,
        'images': {
            'original': img,
            'grayscale': gray,
            'binary_mask': binary_mask,
            'cleaned_mask': cleaned_mask,
            'final_mask': final_mask
        }
    }
    
    return results


def visualize_results(results, save_to_file=True, display=False):
    """
    Visualizes the segmentation pipeline and results.
    
    Args:
        results: Results dictionary from estimate_lesion_coverage
        save_to_file: If True, saves visualization to PNG file
        display: If True, attempts to display plot (may fail in headless environments)
    """
    images = results['images']
    
    # Use non-interactive backend if not displaying
    if not display:
        import matplotlib
        matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lesion Coverage Estimation Pipeline', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(images['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(images['grayscale'], cmap='gray')
    axes[0, 1].set_title('Step 1: Grayscale')
    axes[0, 1].axis('off')
    
    # Binary mask (after Otsu)
    axes[0, 2].imshow(images['binary_mask'], cmap='gray')
    axes[0, 2].set_title("Step 2: Otsu's Thresholding")
    axes[0, 2].axis('off')
    
    # Cleaned mask (after morphological ops)
    axes[1, 0].imshow(images['cleaned_mask'], cmap='gray')
    axes[1, 0].set_title('Step 3: Morphological Ops')
    axes[1, 0].axis('off')
    
    # Final mask (largest component)
    axes[1, 1].imshow(images['final_mask'], cmap='gray')
    axes[1, 1].set_title('Step 4: Largest Component')
    axes[1, 1].axis('off')
    
    # Overlay result
    overlay = images['original'].copy()
    contours, _ = cv2.findContours(images['final_mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Final: Lesion Boundary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save to file if requested
    if save_to_file:
        output_filename = 'lesion_coverage_results.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_filename}")
    
    # Display if requested (may fail in headless environments)
    if display:
        try:
            plt.show()
        except Exception as e:
            print(f"\nNote: Could not display plot (headless environment): {e}")
            print("Plot has been saved to file instead.")
    else:
        plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("LESION COVERAGE ESTIMATION RESULTS")
    print("="*60)
    print(f"Lesion Area:        {results['lesion_area']:,} pixels")
    print(f"Total Image Area:   {results['total_area']:,} pixels")
    print(f"Coverage:           {results['coverage_percentage']:.2f}%")
    print(f"Recommended Patch:  {results['patch_size']}×{results['patch_size']}")
    print(f"Number of Patches:  {results['num_patches']}")
    print(f"\n{results['recommendation']}")
    print("="*60)


def save_results_to_text(results, filename='lesion_coverage_results.txt'):
    """
    Saves results to a text file for record-keeping.
    """
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LESION COVERAGE ESTIMATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Lesion Area:        {results['lesion_area']:,} pixels\n")
        f.write(f"Total Image Area:   {results['total_area']:,} pixels\n")
        f.write(f"Coverage:           {results['coverage_percentage']:.2f}%\n")
        f.write(f"Recommended Patch:  {results['patch_size']}×{results['patch_size']}\n")
        f.write(f"Number of Patches:  {results['num_patches']}\n")
        f.write(f"\n{results['recommendation']}\n")
        f.write("="*60 + "\n")
    
    print(f"✓ Results saved to: {filename}")


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/home/nehal/workspace/cancer_classification/train/BCC/ISIC_0024332.jpg.jpg"

    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Please update the image_path variable with your JPG file path.")
        print(f"Current path '{image_path}' does not exist.")
    else:
        # Process the image
        results = estimate_lesion_coverage(image_path)
        
        # Visualize and save results (no display to avoid Qt issues)
        visualize_results(results, save_to_file=True, display=False)
        
        # Save results to text file
        save_results_to_text(results)
        
        # Save the final mask
        cv2.imwrite('lesion_mask.png', results['images']['final_mask'])
        print(f"✓ Lesion mask saved to: lesion_mask.png")
        
        # Quick summary
        print(f"\n{'='*60}")
        print(f"QUICK SUMMARY:")
        print(f"  Coverage: {results['coverage_percentage']:.2f}%")
        print(f"  Patch Size: {results['patch_size']}×{results['patch_size']}")
        print(f"  Num Patches: {results['num_patches']}")
        print(f"{'='*60}")

# Example usage