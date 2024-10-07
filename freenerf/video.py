import os
from PIL import Image

def create_gif_from_folder(folder_path, output_path, fps=2):
    # Get all image file paths from the specified folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    # Sort the files to ensure the order is correct
    image_files.sort()

    # Check if there are images in the folder
    if not image_files:
        print("No images found in the folder.")
        return

    # Open images and store them in a list
    images = [Image.open(img) for img in image_files]

    # Save as a GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)

# Example usage
folder_path = '/home/thanostriantafyllou/FreeNeRF/DietNeRF-pytorch/dietnerf/logs/ship_16v_full_res_white_bkgd/renderonly_test_049999'
output_path = '/home/thanostriantafyllou/FreeNeRF/DietNeRF-pytorch/dietnerf/logs/ship_16v_full_res_white_bkgd/renderonly_test_049999/animation.gif'
create_gif_from_folder(folder_path, output_path)
