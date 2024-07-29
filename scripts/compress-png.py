import os
import math
from PIL import Image


for target_folder in ['lo-res', 'hi-res']:

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(ROOT, '..', 'figures')
    os.makedirs(os.path.join(OUT_DIR, target_folder, 'compressed'), exist_ok=True)

    Image.MAX_IMAGE_PIXELS = None


    def compress(in_filepath: str, out_filepath: str) -> None:
        img = Image.open(in_filepath)

        total_size = img.size[0] * img.size[1]
        if total_size >= 40000000:
            alpha = math.sqrt((40000000 - 1) / total_size)
            new_size = (int(math.floor(alpha * img.size[0])), int(math.floor(alpha * img.size[1])))
            img.thumbnail(new_size, Image.Resampling.LANCZOS)

        img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        img.save(out_filepath, optimize=True)


    for filename in os.listdir(os.path.join(OUT_DIR, target_folder)):
        in_filepath = os.path.join(OUT_DIR, target_folder, filename)
        out_filepath = os.path.join(OUT_DIR, target_folder, 'compressed', filename)
        if not os.path.isfile(in_filepath):
            continue

        compress(in_filepath, out_filepath)
