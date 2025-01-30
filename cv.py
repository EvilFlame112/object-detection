import cv2
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from pathlib import Path

def process_image(image_path, output_dir):
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    img_mpl = plt.imread(image_path)
    img_cv2 = cv2.imread(image_path)
    
    # Print image shape
    print(f"Image Shape (MPL): {img_mpl.shape}, Image Shape (CV2): {img_cv2.shape}")
    
    # Display pixel distribution
    plt.figure()
    pd.Series(img_mpl.flatten()).plot(kind='hist', bins=50, title='Pixel Distribution')
    pixel_dist_path = output_dir / "pixel_distribution.png"
    plt.savefig(pixel_dist_path)
    plt.close()

    # Show image
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(img_mpl)
    img_display_path = output_dir / "image_display.png"
    plt.savefig(img_display_path)
    plt.close()
    
    # Display RGB channels
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(img_mpl[:, :, 0], cmap='Reds')
    axs[1].imshow(img_mpl[:, :, 1], cmap='Greens')
    axs[2].imshow(img_mpl[:, :, 2], cmap='Blues')
    for ax in axs:
        ax.axis('off')
    rgb_channels_path = output_dir / "rgb_channels.png"
    plt.savefig(rgb_channels_path)
    plt.close()
    
    # Convert BGR to RGB for proper display
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_cv2_rgb)
    ax.axis('off')
    img_cv2_rgb_path = output_dir / "cv2_rgb.png"
    plt.savefig(img_cv2_rgb_path)
    plt.close()
    
    # Blurring the image
    kernel_3x3 = np.ones((3, 3), np.float32) / 10
    blur = cv2.filter2D(img_mpl, -1, kernel_3x3)
    fig, ax = plt.subplots()
    ax.imshow(blur)
    ax.axis('off')
    ax.set_title('Blurred Image')
    blurred_img_path = output_dir / "blurred_image.png"
    plt.savefig(blurred_img_path)
    plt.close()
    
    # Save the blurred image
    output_path = output_dir / "blurred_image.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor((blur * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    return {
        "pixel_distribution": str(pixel_dist_path),
        "image_display": str(img_display_path),
        "rgb_channels": str(rgb_channels_path),
        "cv2_rgb": str(img_cv2_rgb_path),
        "blurred_image": str(blurred_img_path),
        "blurred_image_jpg": str(output_path)
    }
