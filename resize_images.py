import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def resize_image(input_path, output_path, size=(512,512)):
    # Load the image using Pillow
    image = Image.open(input_path)
    
    # Convert the image to RGB or grayscale, depending on its mode
    if image.mode == "L":
        # Grayscale image
        image = image.convert("L")
    else:
        # RGB image
        image = image.convert("RGB")
    
    # Resize the image using OpenCV
    image_array = np.array(image)
    resized_array = cv2.resize(image_array, size, interpolation=cv2.INTER_AREA)
    resized_image = Image.fromarray(resized_array)
    
    # Save the resized image as a new PNG file with a valid sRGB color profile
    with Image.new("RGB", resized_image.size, (255, 255, 255)) as background:
        # Paste the resized image onto the white background to ensure a valid sRGB profile
        if len(resized_image.split()) == 4:  # Check if the image has an alpha channel
            background.paste(resized_image, mask=resized_image.split()[3])
        else:
            background.paste(resized_image, mask=None)
        background.save(output_path, "PNG", icc_profile=open("srgb.icc", "rb").read())


if __name__=="__main__":
    image_dir = Path.cwd()/'train_data/images/'
    output_dir = Path.cwd()/'train_data/resized_images/'
    images_resized_failed = []
    
    # without any multiprocessing
    
    # for image in tqdm(iterable=(image_dir).iterdir(), 
    #                     desc="Resizing images to 512*512",
    #                     total=len(
    #                 list((image_dir).iterdir())
    #             )):
    #     try:
    #         resize_image(image, output_dir/f"{image.stem}.{image.suffix}", size=(512, 512))
    #     except Exception:
    #         images_resized_failed.append(image.stem)

    # with multiprocessing
    images, output_paths, sizes = zip(*[(image, output_dir/f"{image.stem}.{image.suffix}", (512, 512)) for image in image_dir.iterdir()])
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(resize_image, image, output_path, size) for image, output_path, size in zip(images, output_paths, sizes)]
    for future,image in tqdm(zip(as_completed(futures),images), desc="Generating non-resizeable image list.", total=len(images)):
        if future.exception():
            images_resized_failed.append(image.stem)
            
    np.savetxt(output_dir/f"non_resizable_images.txt",
               np.array(images_resized_failed),"%s",
               ',','\n','image_names',encoding='utf-8')