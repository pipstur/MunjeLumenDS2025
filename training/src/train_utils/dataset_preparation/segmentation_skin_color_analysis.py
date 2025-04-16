import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pyrootutils
from keras.layers import Input
from keras.optimizers import Adam
from PIL import Image
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)

import logging

from training.src.models.unet_segmentation import get_unet

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

log = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(model_path, image_size):
    log.info(f"Loading model: <{model_path}>")
    input_img = Input((image_size[0], image_size[1], 3))
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights(model_path)

    return model


def preprocess_image(image_path, image_size):
    """Preprocess the input image for segmentation"""
    # Read and resize image
    img = io.imread(image_path)[:, :, :3]
    img_resized = resize(
        img, (image_size[0], image_size[1], 3), mode="constant", preserve_range=True
    ).astype(np.uint8)

    # Convert to grayscale and apply blackhat
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold and inpaint
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_resized, thresh, 1, cv2.INPAINT_TELEA)

    # Apply Gaussian blur
    final_img = cv2.GaussianBlur(inpainted, (7, 7), 0)

    return img_resized, final_img


def predict_mask(src_path, dest_path, model, image_size):
    """Load image and predict segmentation mask"""
    # Preprocess image
    img_resized, _ = preprocess_image(src_path, image_size)

    # Prepare input for model
    X_test = np.zeros((1, image_size[0], image_size[0], 3), dtype=np.uint8)
    X_test[0] = img_resized

    # Predict
    predicted = model.predict(X_test, verbose=0)
    predicted_mask = (predicted > 0.5).astype(bool).squeeze()

    try:
        mask_img = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        mask_img.save(dest_path)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def batch_predict_mask(
    csv_file: str, image_dir: str, dest_dir: str, model_path: str, image_size: Tuple[int, int]
) -> None:
    """
    Processes a batch of images based on a CSV file and saves the outputs as .png masks.
    """
    df = pd.read_csv(csv_file)
    df["image_name"] = df["image_name"] + ".jpg"

    os.makedirs(dest_dir, exist_ok=True)

    model = load_model(model_path, image_size)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting lesion masks"):
        src_path = os.path.join(image_dir, row["image_name"])

        # Replace .jpg with .png in the destination file name
        base_name = os.path.splitext(row["image_name"])[0]
        dest_path = os.path.join(dest_dir, base_name + ".png")

        predict_mask(src_path, dest_path, model, image_size)

    log.info(f"Predicted masks saved to: <{dest_dir}>")


def classify_skin_tone(ita_val):
    if ita_val > 55:
        return "Very Light"
    elif ita_val > 41:
        return "Light"
    elif ita_val > 28:
        return "Intermediate"
    elif ita_val > 10:
        return "Tan"
    elif ita_val > -30:
        return "Brown"
    else:
        return "Dark"


def predict_skin_color(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Predicts the skin tone category from an RGB image using a binary mask.

    This function extracts skin pixels from the image using the provided mask,
    converts them to the LAB color space, and computes the Individual Typology
    Angle (ITA) to estimate the skin tone.

    Parameters:
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).

    mask : np.ndarray
        Binary mask of shape (H, W). Skin regions should be marked with 0,
        non-skin with 1.
    """
    # Ensure mask has the same size as image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Re-threshold after resizing to ensure binary values (0 or 1)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # Convert whole image to LAB
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Extract LAB values of skin pixels
    skin_lab = image_lab[mask == 0]

    # Compute mean LAB values
    mean_L, mean_A, mean_B = np.mean(skin_lab, axis=0)

    # ITA = arctangent((L - 50) / B) * (180 / pi)
    L = skin_lab[:, 0].astype(np.float32)
    B = skin_lab[:, 2].astype(np.float32)
    ita = np.arctan2((L - 50), B) * 180 / np.pi

    mean_ita = np.mean(ita)

    skin_tone_category = classify_skin_tone(mean_ita)

    return skin_tone_category


def batch_predict_skin_color(
    csv_file: str, image_dir: str, mask_dir: str, output_csv: str = "skin_tone_estimates.csv"
) -> None:
    """
    Processes a batch of images and corresponding masks to estimate skin tone categories.

    This function reads a CSV file containing image names, loads each corresponding image
    and segmentation mask, predicts the skin tone category using `predict_skin_color`, and
    saves the results to a new CSV file.
    """
    df = pd.read_csv(csv_file)
    skin_tones = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting skin tone"):
        base_name = row["image_name"]
        image_path = os.path.join(image_dir, base_name + ".jpg")
        mask_path = os.path.join(mask_dir, base_name + ".png")

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Failed to load image: {image_path}")
            skin_tones.append(None)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Failed to load mask: {mask_path}")
            skin_tones.append(None)
            continue

        try:
            tone = predict_skin_color(image, mask)
        except Exception as e:
            print(f"⚠️ Error processing {base_name}: {e}")
            tone = None

        skin_tones.append(tone)

    df["skin_tone"] = skin_tones
    df.to_csv(output_csv, index=False)
    log.info(f"Predicted skin tones saved to CSV: <{output_csv}>")


def cli():
    parser = argparse.ArgumentParser(
        description="Create masks for lesions and predict skin color."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/test_truth.csv",
        help="Path to the CSV file with labels.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/test_output/train/benign/",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="data/segmentation_mask",
        help="Directory to save masks of lesions.",
    )
    parser.add_argument(
        "--csv-out-path",
        type=str,
        default="data/skin_tone_estimates.csv",
        help="Directory to save CSV with skin color predictions.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/segmentation_model.h5",
        help="Directory to save CSV with skin color predictions.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (width height). Default: 224x224",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli()

    batch_predict_mask(
        args.csv_path, args.images_dir, args.masks_dir, args.model_path, tuple(args.image_size)
    )
    batch_predict_skin_color(args.csv_path, args.images_dir, args.masks_dir, args.csv_out_path)
