import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import os
import re
from pathlib import Path


VISUALIZATION_PATH = Path(__file__).parent.parent.parent / 'visualization'


def sort_key(filename):
    # Extract the number from the filename
    number = re.search(r'\d+', filename)
    if number:
        return int(number.group())
    else:
        return filename


def create_gif(json_id, dir_path):
    # Get all the PNG files with the specified ID
    filenames = glob.glob(f"{dir_path}/*_wigner_{json_id}.png")
    # Sort the filenames numerically
    filenames = sorted(filenames, key=sort_key)
    # Read the images into a list
    images = [PILImage.open(filename) for filename in filenames]

    # Save the images as a GIF
    images[0].save(VISUALIZATION_PATH / dir_path / f"wigner_evolution_{json_id}.gif",
                   save_all=True,
                   append_images=images[1:],
                   optimize=False,
                   duration=100,
                   loop=0)

    # Delete the PNG files
    for filename in filenames:
        os.remove(filename)

    return f"wigner_evolution_{json_id}.gif"