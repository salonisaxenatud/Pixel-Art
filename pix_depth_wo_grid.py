import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import colorsys
from sklearn.cluster import KMeans

def extract_depth(content_img, model_type="DPT_Large"):
    """
    Extract depth information from an image using MiDaS depth estimation model.
    
    Args:
        content_img: numpy array, Input image
        model_type: str, MiDaS model variant to use ('DPT_Large', 'DPT_Hybrid', or other)
        
    Returns:
        torch.Tensor: Normalized depth map
    """
    # Load MiDaS model and move to appropriate device
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    # Get appropriate transform based on model type
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Process image and get depth prediction
    input_batch = transform(content_img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=content_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalize depth values to [0,1] range
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = torch.tensor(depth_map, dtype=torch.float32)

    return depth_map

class DepthPixelArtProcessor:
    """
    A class for creating depth-aware pixel art from images.
    Uses depth information to apply different levels of pixelation.
    """

    def __init__(self, image, depth_map):
        """
        Initialize the processor with an image and its depth map.
        
        Args:
            image: numpy array, Input image in RGB format
            depth_map: numpy array, Depth map for the image
        """
        self.original_image = image
        self.depth_map = depth_map
        self.height, self.width = image.shape[:2]
        
        # Normalize depth map and create masks for different depth ranges
        self.depth_map_normalized = cv2.normalize(self.depth_map, None, alpha=0, beta=1, 
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Create boolean masks for foreground, middle ground, and background based on depth thresholds
        self.foreground_mask = self.depth_map_normalized > (2.0/3)
        self.middle_ground_mask = (self.depth_map_normalized <= (2.0/3)) & (self.depth_map_normalized > (1.0/3))
        self.background_mask = self.depth_map_normalized <= (1.0/3)
        
        # Define preset color palettes for different visual styles
        self.pixel_art_palettes = {
            'gameboy': np.array([
                [15, 56, 15], [48, 98, 48],
                [139, 172, 15], [155, 188, 15]
            ]),
            'nostalgia': np.array([
                [0, 0, 0], [255, 255, 255], [255, 0, 0],
                [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 128, 0], [128, 0, 128]
            ]),
            'db16': np.array([  # DawnBringer's 16 color palette
                [20, 12, 28],    # Dark purple
                [68, 36, 52],    # Dark red
                [48, 52, 109],   # Navy blue
                [78, 74, 78],    # Gray
                [133, 76, 48],   # Brown
                [52, 101, 36],   # Dark green
                [208, 70, 72],   # Red
                [117, 113, 97],  # Beige
                [89, 125, 206],  # Light blue
                [210, 125, 44],  # Orange
                [133, 149, 161], # Light gray
                [109, 170, 44],  # Green
                [210, 170, 153], # Light peach
                [109, 194, 202], # Cyan
                [218, 212, 94],  # Yellow
                [222, 238, 214]  # White
            ])
        }

    def create_color_ramp(self, base_color, steps=2):
        """
        Create variations of a base color by adjusting its saturation and value.
        
        Args:
            base_color: numpy array, Base RGB color
            steps: int, Number of variations to create
            
        Returns:
            numpy array: Array of color variations
        """
        h, s, v = colorsys.rgb_to_hsv(base_color[0] / 255, base_color[1] / 255, base_color[2] / 255)
        ramp = []
        for i in range(steps):
            new_v = max(0, v - (0.2 * (steps - i - 1)))
            new_s = min(1, s + (0.1 * (steps - i - 1)))
            rgb = colorsys.hsv_to_rgb(h, new_s, new_v)
            ramp.append(np.array([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]))
        return np.array(ramp)

    def generate_adaptive_palette(self, n_base_colors=8, shades_per_color=4):
        """
        Generate a color palette based on the dominant colors in the image.
        
        Args:
            n_base_colors: int, Number of base colors to extract
            shades_per_color: int, Number of shades to generate for each base color
            
        Returns:
            numpy array: Generated color palette
        """
        pixels = self.original_image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_base_colors, random_state=42)
        kmeans.fit(pixels)
        base_colors = kmeans.cluster_centers_
        palette = []
        for base_color in base_colors:
            ramp = self.create_color_ramp(base_color, shades_per_color)
            palette.extend(ramp)
        return np.array(palette)

    def quantize_to_palette(self, image, palette):
        """
        Map each pixel in the image to the nearest color in the palette.
        
        Args:
            image: numpy array, Input image
            palette: numpy array, Target color palette
            
        Returns:
            numpy array: Color-quantized image
        """
        h, w = image.shape[:2]
        image_reshaped = image.reshape(-1, 3)
        distances = np.sqrt(np.sum((image_reshaped[:, np.newaxis] - palette) ** 2, axis=2))
        nearest_idx = np.argmin(distances, axis=1)
        quantized = palette[nearest_idx].reshape(h, w, 3)
        return quantized.astype(np.uint8)

    def apply_depth_pixelation(self, small_size=5, medium_size=10, large_size=20, 
                             palette_name=None, custom_palette=None, adaptive_palette=False):
        """
        Apply depth-aware pixelation to create pixel art with varying pixel sizes.
        
        Args:
            small_size: int, Pixel size for foreground objects
            medium_size: int, Pixel size for middleground objects
            large_size: int, Pixel size for background objects
            palette_name: str, Name of predefined palette to use
            custom_palette: numpy array, Custom color palette
            adaptive_palette: bool, Whether to generate palette from image colors
            
        Returns:
            numpy array: Processed pixel art image
        """
        # Select appropriate color palette based on input parameters
        if palette_name and palette_name in self.pixel_art_palettes:
            palette = self.pixel_art_palettes[palette_name]
        elif custom_palette is not None:
            palette = custom_palette
        elif adaptive_palette:
            palette = self.generate_adaptive_palette()
        else:
            palette = self.pixel_art_palettes['nostalgia']

        # Create a copy of the original image for the result
        result = self.original_image.copy()
        
        # Process each depth layer with its corresponding pixel size
        for mask, size in [(self.foreground_mask, small_size), 
                          (self.middle_ground_mask, medium_size), 
                          (self.background_mask, large_size)]:
            # Apply pixelation effect by downscaling and upscaling
            pixelated = cv2.resize(self.original_image, (0, 0), 
                                 fx=1/size, fy=1/size, 
                                 interpolation=cv2.INTER_NEAREST)
            pixelated = cv2.resize(pixelated, (self.width, self.height), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Apply color palette quantization
            pixelated = self.quantize_to_palette(pixelated, palette)
            
            # Combine the pixelated region with the result using the depth mask
            result = np.where(mask[..., None], pixelated, result)
        
        return result

    def display_image(self, image):
        """
        Display the processed image using matplotlib.
        
        Args:
            image: numpy array, Image to display
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title('Depth-Based Pixel Art')
        plt.axis('off')
        plt.show()

def main():
    """
    Main function for demonstrating the depth-based pixel art generation process.
    Loads an image, processes it, and saves multiple versions with different palettes.
    """
    image_path = "./cow.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert image from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract depth information from the image
    depth_tensor = extract_depth(image)
    depth_array = depth_tensor.squeeze().cpu().numpy()

    # Create processor instance and generate different versions
    processor = DepthPixelArtProcessor(image, depth_array)

    # Process image with db16 palette
    result1 = processor.apply_depth_pixelation(palette_name='db16')
    cv2.imwrite("pixel_art/depth_based/outputs/cow_db16.jpg", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    
    # Process image with adaptive palette
    result2 = processor.apply_depth_pixelation(adaptive_palette=True)
    cv2.imwrite("pixel_art/depth_based/outputs/cow_adaptive.jpg", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))

    # Display results in a comparative view
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(result1)
    plt.title("db16 Palette")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(result2)
    plt.title("Adaptive Palette")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()