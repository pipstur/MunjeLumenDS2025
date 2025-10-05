from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.filters import frangi


def resize_image(
    image: Image.Image,
    image_size: Tuple[int, int],
    padding_flag: bool,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Resizes an image to the target size, with optional padding.

    Args:
        image (Image.Image): Input image.
        image_size (Tuple[int, int]): Desired (width, height) of the output image.
        padding_flag (bool): If True, maintains aspect ratio and pads the image.
        padding_color (Tuple[int, int, int], optional): Color used for padding. Defaults to white.

    Returns:
        Image.Image: The resized (and optionally padded) image.
    """
    if padding_flag:
        canvas_size = image_size
        image.thumbnail(canvas_size, Image.LANCZOS)
        canvas = Image.new("RGB", canvas_size, padding_color)
        paste_x = (canvas_size[0] - image.size[0]) // 2
        paste_y = (canvas_size[1] - image.size[1]) // 2
        canvas.paste(image, (paste_x, paste_y))
        image = canvas
    else:
        image = image.resize(image_size, Image.BILINEAR)

    return image


def apply_clahe(
    image: Image.Image, clip_limit: float = 1.0, tile_grid_size: tuple = (5, 5)
) -> Image.Image:
    """
    Enhances the contrast of an image using CLAHE
    (Contrast Limited Adaptive Histogram Equalization)
    applied to the lightness (L) channel of the LAB color space.
    CLAHE is applied only to the L channel to enhance contrast without altering colors

    Parameters:
    - image (PIL.Image.Image): Input image in RGB format.
    - clip_limit (float, optional): Threshold for contrast limiting. Higher
     values increase contrast but may amplify noise. Default is 1.0.
    - tile_grid_size (tuple of int, optional): Size of the grid for dividing the image into tiles
      (e.g., (5, 5)). CLAHE is applied to each tile individually, allowing
      localized contrast enhancement.
      Smaller tiles result in more localized adjustments. Default is (5, 5).

    Returns:
    - PIL.Image.Image: The contrast-enhanced image in RGB format.
    """

    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    enhanced_pil = Image.fromarray(enhanced_image)

    return enhanced_pil


def remove_hair(
    image: Image.Image,
    kernel_size: Tuple[int, int] = (9, 9),
    blur_size: Tuple[int, int] = (3, 3),
    threshold: int = 10,
    inpaint_radius: int = 3,
    inpaint_method: int = cv2.INPAINT_TELEA,
) -> Image.Image:
    """
    Removes hair artifacts from a PIL image using morphological operations and inpainting.

    The process involves:
    - Converting the image to grayscale.
    - Applying a black hat morphological operation to highlight hair strands.
    - Blurring the result to reduce noise.
    - Creating a binary mask by thresholding the blurred image.
    - Using inpainting to remove the hair from the original image based on the mask.

    Args:
        image (PIL.Image.Image): Input image in RGB format.
        kernel_size (Tuple[int, int], optional): Size of the structuring element for blackhat
        blur_size (Tuple[int, int], optional): Kernel size for Gaussian blur. Must be odd.
        threshold (int, optional): Threshold value for binary mask creation.
        inpaint_radius (int, optional): Radius of the circular neighborhood for inpainting.
        inpaint_method (int, optional): OpenCV inpainting method
    Returns:
        PIL.Image.Image: A new image with hair artifacts removed.
    """
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(blackhat, blur_size, 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image_np, mask, inpaint_radius, inpaint_method)
    return Image.fromarray(cleaned)


def remove_hair_frangi(
    image: Image.Image,
    inpaint_radius: int = 3,
    inpaint_method: int = cv2.INPAINT_TELEA,
    threshold: int = 0.1,
) -> Image.Image:
    """
    Hair removal using Frangi filter for vessel-like (hair) detection.
    """
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    frangi_response = frangi(gray)
    frangi_norm = (frangi_response * 255 / frangi_response.max()).astype(np.uint8)

    _, mask = cv2.threshold(frangi_norm, int(threshold * 255), 255, cv2.THRESH_BINARY)

    cleaned = cv2.inpaint(image_np, mask, inpaint_radius, inpaint_method)
    return Image.fromarray(cleaned)
