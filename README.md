# Pixel-Art
 
<div style="text-align: center; gap: 50px;">
    <img src="cow.jpg" alt="Content mesh" width="30%" height="30%">
    <img src="outputs/pixelated_cow_db16_100px.jpg" width="30%" alt="Style image" height="30%">
</div>
This feature enables users to apply a pixel art effect to any image, leveraging classic image processing techniques to authentically replicate the distinctive style of this art form.  

## Key Features  

- **Classic Pixel Art Rendering** – Achieved through downsampling and k-nearest neighbors (KNN) search to create a simplified, pixelated representation of the original image.  
- **Adaptive Palette and Quantization** – Since pixel art typically utilizes a limited color palette, this feature extracts the most dominant colors from the image and reduces them to a refined, cohesive palette.  
- **Edge Detection and Enhancement** – Black outlines, a prominent feature of pixel art, are generated using edge detection techniques to either fully outline objects or emphasize key features.  
- **Real-Time Application** – The script can be adapted for real-time video processing, allowing pixel art effects to be applied dynamically to live video feeds.  

# Depth Based Pixel Art
<div style="text-align: center; gap: 50px;">
    <img src="cow.jpg" alt="Content mesh" width="30%" height="30%">
    <img src="outputs/depth_based/outputs/cow_db16.jpg" width="30%" alt="Style image" height="30%">

**Pipeline**:

<img src = "outputs/pipeline_pixdepth.jpg" width = "80%" >

- For Depth Map Extraction [MiDas](https://github.com/isl-org/MiDaS) was used for
monocular depth estimation

<img src="outputs/depth_based/outputs/cow_depth.jpg" width = "30%">

- Image is then segmented into 3 layer Masks. 

<div style="text-align: center; gap: 50px;">
    <img src="outputs/depth_based/outputs/cow_bg.jpg" width="25%" height="25%">
    <img src="outputs/depth_based/outputs/cow_mg.jpg" width="25%" height="25%">
    <img src="outputs/depth_based/outputs/cow_fg.jpg" width="25%" height="25%">


- Further, grid-based segmentation ensures alignment with pixelation boundaries. 

(Left: Edges on simply merging masks; Right: Edges after specifically processing)

<div style="text-align: center; gap: 50px;">
    <img src="outputs/depth_based/outputs_wo_grid/wogrid_ss.png" width="30%" height="30%">
    <img src="outputs/depth_based/outputs/Screenshot 2025-01-30 111824.png" width="30%" height="30%">

- Pixel size varies by depth. Each block’s average color is quantized to the closest palette color.
- Initial approach included merging layers sequentially, leading to possible artifacts and overlapping at depth boundaries.

<img src = "outputs/depth_based/outputs_wo_grid/overlap.png" width = "20%">

- Merged masks in an edge-conscious way. The grid structure assigns blocks to depth layers and ensures clean, grid-aligned boundaries. Whole blocks are included in a mask if any pixel falls in that depth layer.

(Top row (from left): foreground, middleground, and background; Bottom row: After resolving mask conflict)
<img src="outputs/combined.jpg" width = "80%">

