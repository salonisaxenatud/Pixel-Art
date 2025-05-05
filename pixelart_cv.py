import numpy as np
import cv2
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt

class PixelArtStyleTransfer:
    """
    A class for transforming regular images into pixel art style using color quantization
    and optional outline enhancement.
    
    Features:
    - Multiple predefined color palettes (gameboy, nostalgia, db16)
    - Adaptive palette generation from image colors
    - Custom palette support
    - Edge detection and enhancement
    """

    def __init__(self, image=None):
        """
        Initialize the PixelArtStyleTransfer with an input image.
        
        Args:
            image: numpy array, Input image in BGR or RGB format
        """
        if image is not None:
            # Convert input image to RGB if necessary
            if len(image.shape) == 3:
                self.original_image = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                self.original_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.current_image = self.original_image.copy()
            self.height, self.width = self.original_image.shape[:2]
        else:
            raise ValueError("An initial image is required.")
        
        # Define classic pixel art color palettes
        self.pixel_art_palettes = {
            'gameboy': np.array([
                [15, 56, 15],    # Darkest green
                [48, 98, 48],    # Dark green
                [139, 172, 15],  # Light green
                [155, 188, 15]   # Lightest green
            ]),
            'nostalgia': np.array([
                [0, 0, 0],        # Black
                [255, 255, 255],  # White
                [255, 0, 0],      # Red
                [0, 255, 0],      # Green
                [0, 0, 255],      # Blue
                [255, 255, 0],    # Yellow
                [255, 128, 0],    # Orange
                [128, 0, 128]     # Purple
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

    def update_image(self, image):
        """
        Update the current working image for real-time processing.
        
        Args:
            image: numpy array, New image in BGR format
        """
        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape[:2]

    def create_color_ramp(self, base_color, steps=2):
        """
        Create a gradient of colors from a base color by adjusting saturation and value.
        
        Args:
            base_color: numpy array, Base RGB color to create variations from
            steps: int, Number of color variations to generate
            
        Returns:
            numpy array: Array of colors forming a gradient
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
        Generate a color palette adapted to the image's dominant colors using K-means clustering.
        
        Args:
            n_base_colors: int, Number of base colors to extract
            shades_per_color: int, Number of shades to generate for each base color
            
        Returns:
            numpy array: Generated color palette
        """
        pixels = self.current_image.reshape(-1, 3)
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
            palette: numpy array, Color palette to quantize to
            
        Returns:
            numpy array: Color-quantized image
        """
        h, w = image.shape[:2]
        image_reshaped = image.reshape(-1, 3)
        distances = np.sqrt(np.sum((image_reshaped[:, np.newaxis] - palette) ** 2, axis=2))
        nearest_idx = np.argmin(distances, axis=1)
        quantized = palette[nearest_idx].reshape(h, w, 3)
        return quantized.astype(np.uint8)

    def detect_and_enhance_outlines(self, thickness=1, threshold=200):
        """
        Detect edges using Sobel operators and enhance them with black outlines.
        
        Args:
            thickness: int, Thickness of the outline
            threshold: int, Edge detection sensitivity threshold
            
        Returns:
            numpy array: Image with enhanced outlines
        """
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edges = (magnitude > threshold).astype(np.uint8) * 255
        
        if thickness > 1:
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
        outline_mask = edges > 0
        result = self.current_image.copy()
        result[outline_mask] = [0, 0, 0]  # Black outlines
        self.current_image = result
        return self.current_image

    def apply_pixel_art_style(self, target_pixels=32, palette_name=None, custom_palette=None, skip_outlines=False):
        """
        Transform the image into pixel art style.
        
        Args:
            target_pixels: int, Target resolution
            palette_name: str, Name of predefined palette to use
            custom_palette: numpy array, Optional custom color palette
            skip_outlines: bool, Whether to skip outline enhancement
            
        Returns:
            numpy array: Processed pixel art image
        """
        # Calculate scaling factor for pixelation
        factor = max(1, min(self.height, self.width) // target_pixels)
        h, w = self.height // factor, self.width // factor
        
        # Process entire image
        small = cv2.resize(self.original_image, (w, h), interpolation=cv2.INTER_AREA)
        
        # Select and apply color palette
        if palette_name and palette_name in self.pixel_art_palettes:
            palette = self.pixel_art_palettes[palette_name]
        elif custom_palette is not None:
            palette = custom_palette
        else:
            palette = self.generate_adaptive_palette()
            
        pixelated = self.quantize_to_palette(small, palette)
        
        # Add outlines if not skipped
        if not skip_outlines:
            self.current_image = pixelated
            pixelated = self.detect_and_enhance_outlines()
            
        result = cv2.resize(pixelated, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        self.current_image = result
        return result

    def extract_palette_from_image(self, palette_image_path, n_colors=8):
        """
        Extract a color palette from a reference image using K-means clustering.
        
        Args:
            palette_image_path: str, Path to the reference image
            n_colors: int, Number of colors to extract
            
        Returns:
            numpy array: Extracted color palette
        """
        palette_image = cv2.imread(palette_image_path)
        palette_image = cv2.cvtColor(palette_image, cv2.COLOR_BGR2RGB)
        pixels = palette_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(pixels)
        palette = kmeans.cluster_centers_.astype(int)
        return palette

def main():
  
    image_path = "./cow.jpg" 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processor = PixelArtStyleTransfer(image)

    adaptive_result = processor.apply_pixel_art_style(target_pixels=100)  # Uses adaptive palette by default

    # Display results
    plt.figure(figsize=(15, 30))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adaptive_result)
    plt.title("Adaptive Palette")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------
# Uncomment for real time using your web camera
# ------------------------------------------------------------------------------------

# def main():

#     cap = cv2.VideoCapture(0)
#     processor = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if processor is None:
#             processor = PixelArtStyleTransfer(frame)
#         else:
#             processor.update_image(frame)

#         processed_frame = processor.apply_pixel_art_style(target_pixels=100, palette_name='db16')
#         cv2.imshow("Pixel Art Style Transfer", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
