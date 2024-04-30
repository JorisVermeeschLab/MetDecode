import os
from PIL import Image


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, '..', 'figures')
os.makedirs(os.path.join(OUT_DIR, 'lo-res', 'compressed'), exist_ok=True)

Image.MAX_IMAGE_PIXELS = None


def compress(in_filepath, out_filepath):
    img = Image.open(in_filepath)
    img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
    img.save(out_filepath, optimize=True)


for filename in os.listdir(os.path.join(OUT_DIR, 'lo-res')):
    in_filepath = os.path.join(OUT_DIR, 'lo-res', filename)
    out_filepath = os.path.join(OUT_DIR, 'lo-res', 'compressed', filename)
    if not os.path.isfile(in_filepath):
        continue

    compress(in_filepath, out_filepath)
