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
        
    Raises:
        ValueError: If model prediction or depth map conversion fails
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
        # Resize prediction to match input image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=content_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Validate and normalize depth map
    if not isinstance(prediction, torch.Tensor):
        raise ValueError("Model prediction is not a tensor")

    depth_map = prediction.cpu().numpy()
    if not isinstance(depth_map, np.ndarray):
        raise ValueError("depth_map must be a NumPy array")

    # Normalize depth values to [0,1] range
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = torch.tensor(depth_map, dtype=torch.float32)

    return depth_map

class DepthPixelArtProcessor:
    """
    A class for creating depth-aware pixel art from images.
    Uses depth information to apply different levels of pixelation,
    creating a pseudo-3D effect in the final pixel art.
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
        
        # Normalize depth map to [0,1] range and resize to match image
        self.depth_map_normalized = cv2.normalize(self.depth_map, None, alpha=0, beta=1, 
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.depth_map_normalized = cv2.resize(self.depth_map_normalized, 
                                             (self.width, self.height), 
                                             interpolation=cv2.INTER_LINEAR)
        
        # Define preset color palettes for pixel art
        self.pixel_art_palettes = {
            'gameboy': np.array([[15, 56, 15], [48, 98, 48],
                               [139, 172, 15], [155, 188, 15]]),
            'nostalgia': np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                                  [0, 255, 0], [0, 0, 255], [255, 255, 0],
                                  [255, 128, 0], [128, 0, 128]]),
            'db16': np.array([  # DawnBringer's 16 color palette
                [20, 12, 28], [68, 36, 52], [48, 52, 109], [78, 74, 78],
                [133, 76, 48], [52, 101, 36], [208, 70, 72], [117, 113, 97],
                [89, 125, 206], [210, 125, 44], [133, 149, 161], [109, 170, 44],
                [210, 170, 153], [109, 194, 202], [218, 212, 94], [222, 238, 214]
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
        
        # Debug information for shape mismatches
        print(f"image_reshaped shape: {image_reshaped.shape}")
        print(f"distances shape: {distances.shape}")
        print(f"nearest_idx shape: {nearest_idx.shape}")
        print(f"palette shape: {palette.shape}")
        
        quantized = palette[nearest_idx].reshape(h, w, 3)
        return quantized.astype(np.uint8)
    
    def create_grid_masks(self, pixel_size):
        """
        Create depth-based masks for foreground, middleground, and background.
        
        Args:
            pixel_size: int, Size of pixelation grid cells
            
        Returns:
            tuple: Three boolean masks for foreground, middleground, and background
        """
        grid_h = (self.height + pixel_size - 1) // pixel_size
        grid_w = (self.width + pixel_size - 1) // pixel_size
        
        # Create grid coordinates
        y_coords = np.repeat(np.arange(grid_h), grid_w)
        x_coords = np.tile(np.arange(grid_w), grid_h)
        
        # Calculate average depth for each grid cell
        grid_depths = np.zeros((grid_h, grid_w))
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * pixel_size
                y_end = min((i + 1) * pixel_size, self.height)
                x_start = j * pixel_size
                x_end = min((j + 1) * pixel_size, self.width)
                
                cell_depth = self.depth_map_normalized[y_start:y_end, x_start:x_end].mean()
                grid_depths[i, j] = cell_depth
        
        # Create masks for different depth ranges
        grid_fg = grid_depths > (2.0/3) # foreground
        grid_mg = (grid_depths <= (2.0/3)) & (grid_depths > (1.0/3)) # middleground
        grid_bg = grid_depths <= (1.0/3) # background
        
        # Resize masks to match image dimensions
        fg_mask = np.repeat(np.repeat(grid_fg, pixel_size, axis=0), pixel_size, axis=1)
        mg_mask = np.repeat(np.repeat(grid_mg, pixel_size, axis=0), pixel_size, axis=1)
        bg_mask = np.repeat(np.repeat(grid_bg, pixel_size, axis=0), pixel_size, axis=1)
        
        # Crop masks to exact image dimensions
        fg_mask = fg_mask[:self.height, :self.width]
        mg_mask = mg_mask[:self.height, :self.width]
        bg_mask = bg_mask[:self.height, :self.width]
        
        return fg_mask, mg_mask, bg_mask

    def get_block_average_color(self, image, y_start, y_end, x_start, x_end):
        """
        Calculate the average color of a block of pixels.
        
        Args:
            image: numpy array, Source image
            y_start, y_end, x_start, x_end: int, Block coordinates
            
        Returns:
            numpy array: Average RGB color of the block
        """
        block = image[y_start:y_end, x_start:x_end]
        return np.mean(block, axis=(0, 1))

    def quantize_block(self, avg_color, palette):
        """
        Find the nearest palette color for a given color.
        
        Args:
            avg_color: numpy array, Input color to quantize
            palette: numpy array, Target color palette
            
        Returns:
            numpy array: Nearest palette color
        """
        distances = np.sqrt(np.sum((avg_color - palette) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        return palette[nearest_idx]

    def create_aligned_mask(self, depth_mask, pixel_size):
        """
        Create a mask that aligns with pixel boundaries.
        
        Args:
            depth_mask: numpy array, Input depth mask
            pixel_size: int, Size of pixels
            
        Returns:
            numpy array: Boolean mask aligned to pixel boundaries
        """
        grid_h = (self.height + pixel_size - 1) // pixel_size
        grid_w = (self.width + pixel_size - 1) // pixel_size
        
        aligned_mask = np.zeros((self.height, self.width), dtype=bool)
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * pixel_size
                y_end = min((i + 1) * pixel_size, self.height)
                x_start = j * pixel_size
                x_end = min((j + 1) * pixel_size, self.width)
                
                block_mask = depth_mask[y_start:y_end, x_start:x_end]
                if block_mask.any():
                    aligned_mask[y_start:y_end, x_start:x_end] = True
                    
        return aligned_mask

    def resolve_mask_conflicts(self, fg_mask, mg_mask, bg_mask, small_size, medium_size, large_size):
        """
        Resolve overlaps between depth masks, prioritizing smaller pixels.
        
        Args:
            fg_mask, mg_mask, bg_mask: numpy arrays, Depth layer masks
            small_size, medium_size, large_size: int, Pixel sizes for each depth layer
            
        Returns:
            tuple: Three non-overlapping boolean masks
        """
        fg_aligned = self.create_aligned_mask(fg_mask, small_size)
        mg_aligned = self.create_aligned_mask(mg_mask, medium_size)
        bg_aligned = self.create_aligned_mask(bg_mask, large_size)
        
        # Remove overlaps, prioritizing smaller pixels
        mg_aligned[fg_aligned] = False
        bg_aligned[fg_aligned | mg_aligned] = False
        
        return fg_aligned, mg_aligned, bg_aligned

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
        # Select color palette
        if palette_name and palette_name in self.pixel_art_palettes:
            palette = self.pixel_art_palettes[palette_name]
        elif custom_palette is not None:
            palette = custom_palette
        elif adaptive_palette:
            palette = self.generate_adaptive_palette()
        else:
            palette = self.pixel_art_palettes['nostalgia']

        result = np.zeros_like(self.original_image)
        
        # Create and resolve depth masks
        fg_mask, mg_mask, bg_mask = self.create_grid_masks(small_size)
        fg_final, mg_final, bg_final = self.resolve_mask_conflicts(
            fg_mask, mg_mask, bg_mask, small_size, medium_size, large_size)

        # Process each depth layer with appropriate pixel size
        for size, mask in [(small_size, fg_final), 
                          (medium_size, mg_final), 
                          (large_size, bg_final)]:
            grid_h = (self.height + size - 1) // size
            grid_w = (self.width + size - 1) // size
            
            for i in range(grid_h):
                for j in range(grid_w):
                    y_start = i * size
                    y_end = min((i + 1) * size, self.height)
                    x_start = j * size
                    x_end = min((j + 1) * size, self.width)
                    
                    # Check if this block is part of the current depth layer
                    block_mask = mask[y_start:y_end, x_start:x_end]
                    if block_mask.any():
                        # Get average color for the block
                        avg_color = self.get_block_average_color(
                            self.original_image, y_start, y_end, x_start, x_end)
                        
                        # Quantize the average color to the palette
                        quantized_color = self.quantize_block(avg_color, palette)
                        
                        # Fill the block with the quantized color
                        result[y_start:y_end, x_start:x_end][block_mask] = quantized_color
        
        # Fill any remaining black pixels with background quantized colors
        black_pixels = np.all(result == 0, axis=2)
        if black_pixels.any():
            bg_quantized = self.quantize_to_palette(self.original_image, palette)
            result[black_pixels] = bg_quantized[black_pixels]
        
        return result

    def display_image(self, image):
        """
        Display image using Matplotlib.
        
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
    Main function for running the depth-based pixel art generation process. This was used for testing the script locally.
    
    """
    image_path = "./cow.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth_tensor = extract_depth(image)
    depth_array = depth_tensor.squeeze().cpu().numpy()
    cv2.imwrite("pixel_art/depth_based/outputs/cow_depth.jpg", (depth_array * 255).astype(np.uint8))

    processor = DepthPixelArtProcessor(image, depth_array)

    result1 = processor.apply_depth_pixelation(palette_name='db16')
    cv2.imwrite("pixel_art/depth_based/outputs/cow_db16.jpg", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
    result2 = processor.apply_depth_pixelation(adaptive_palette=True)
    cv2.imwrite("pixel_art/depth_based/outputs/cow_adaptive.jpg", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))

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