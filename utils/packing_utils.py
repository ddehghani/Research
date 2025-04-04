import math
from PIL import Image
import rpack
import numpy as np
from models import Instance
from .image_utils import add_padding

def pack(images: list, grid_width: int, grid_height: int, padding_width: float, padding_color: int):
    """
    Packs a list of images into a compact layout using rectangle packing, adding padding around each image.

    Parameters:
    - images: list of paths to image files
    - grid_width: base grid width for determining layout scaling
    - grid_height: base grid height for determining layout scaling
    - padding_width: amount of padding (in pixels) to add around each image
    - padding_color: RGB value to use for the padding color

    Returns:
    - annotations: list of Instance objects with bounding boxes for each image
    - packed_image: a single PIL.Image with all input images packed together
    """
    padded_sizes = []
    total_area = 0

    for img_path in images:
        img = Image.open(img_path)
        width, height = img.size
        width += 2 * padding_width
        height += 2 * padding_width
        total_area += width * height
        padded_sizes.append((width, height))

    total_area *= 1.7
    ratio = math.ceil(math.sqrt(total_area / (grid_width * grid_height)))
    packed_positions = rpack.pack(padded_sizes, grid_width * ratio)

    max_width = max(pos[0] + size[0] for pos, size in zip(packed_positions, padded_sizes))
    max_height = max(pos[1] + size[1] for pos, size in zip(packed_positions, padded_sizes))

    annotations = []
    packed_image = Image.new('RGB', (max_width, max_height))

    for idx, img_path in enumerate(images):
        img = Image.open(img_path)
        padded_img = add_padding(img, padding_width, padding_width, padding_width, padding_width, padding_color)
        annotations.append(Instance(
            name=img_path,
            confidence=1,
            bbox=np.array([
                packed_positions[idx][0] + padding_width,
                packed_positions[idx][1] + padding_width,
                img.width,
                img.height
            ])
        ))
        packed_image.paste(padded_img, packed_positions[idx])

    return annotations, packed_image

def gridify(images: list, grid_width: int, grid_height: int, padding_width: float, padding_color: int):
    """
    Arranges a list of images into a fixed-size grid, adding equal padding around each image to standardize dimensions.

    Parameters:
    - images: list of paths to image files
    - grid_width: number of columns in the output grid
    - grid_height: number of rows in the output grid
    - padding_width: amount of padding (in pixels) to add around each image
    - padding_color: RGB value to use for the padding color

    Returns:
    - annotations: list of Instance objects with bounding boxes for each image in the grid
    - grid_image: a single PIL.Image showing all input images arranged in a grid
    """
    max_width, max_height = 0, 0
    for img_path in images:
        img = Image.open(img_path)
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)

    max_width += 2 * padding_width
    max_height += 2 * padding_width

    annotations = []
    grid_image = Image.new('RGB', (grid_width * max_width, grid_height * max_height))

    for idx, img_path in enumerate(images):
        img = Image.open(img_path)
        top = (max_height - img.height) // 2
        right = (max_width - img.width) // 2
        padded_img = add_padding(img, top, right, top, right, padding_color)
        row, col = divmod(idx, grid_width)
        annotations.append(Instance(
            name=img_path.split('.')[0].split('_')[-1],
            confidence=1,
            bbox=np.array([
                col * max_width + right,
                row * max_height + top,
                img.width,
                img.height
            ])
        ))
        grid_image.paste(padded_img, (col * max_width, row * max_height))

    return annotations, grid_image