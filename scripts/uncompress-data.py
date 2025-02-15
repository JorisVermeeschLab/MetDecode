import os
import zipfile


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, "data")

for folder in ["30_50bp", "30_100bp", "30_250bp"]:
    for subfolder in ["all-markers"]:
        for filename in ["all", "all-insilico", "cfdna"]:
            in_filepath = os.path.join(DATA, folder, subfolder, f"{filename}.zip")
            with zipfile.ZipFile(in_filepath, 'r') as z:
                z.extract(f"{filename}.tsv", os.path.join(DATA, folder, subfolder))
