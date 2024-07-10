from PIL import Image
import os

def modify_background(image_path, output_path, background_color=(0, 0, 255)):
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new('RGB', img.size, background_color)
            background.paste(img, (0, 0), img)
            background.save(output_path)
        else:
            img.save(output_path)

def process_images(source_dir, target_dir, background_color=(0, 0, 255)):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                image_file = os.path.join(root, file)
                output_file = image_file.replace(source_dir, target_dir)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                modify_background(image_file, output_file, background_color)
