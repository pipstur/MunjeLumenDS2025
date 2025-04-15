import os

import cv2
import numpy as np
import pandas as pd


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


def predict_skin_color(image, mask):
    # Ensure mask has the same size as image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Re-threshold after resizing to ensure binary values (0 or 1)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # skin_pixels = image[mask == 0]  # shape: (N_skin_pixels, 3)

    # Convert whole image to LAB
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Extract LAB values of skin pixels
    skin_lab = image_lab[mask == 0]

    # Compute mean LAB values
    mean_L, mean_A, mean_B = np.mean(skin_lab, axis=0)

    # ITA = arctangent((L - 50) / B) in degrees
    L = skin_lab[:, 0].astype(np.float32)
    B = skin_lab[:, 2].astype(np.float32)
    ita = np.arctan2((L - 50), B) * 180 / np.pi

    mean_ita = np.mean(ita)

    skin_tone_category = classify_skin_tone(mean_ita)

    return skin_tone_category


def process_image_batch(
    csv_file: str, image_dir: str, mask_dir: str, output_csv: str = "skin_tone_estimates.csv"
) -> None:
    """Loads images and masks, predicts skin tone category, and writes result to a new CSV."""
    df = pd.read_csv(csv_file)

    skin_tones = []

    for _, row in df.iterrows():
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

    # Append results to the dataframe and save
    df["skin_tone"] = skin_tones
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved updated CSV to {output_csv}")


CSV_PATH = "/Users/mipopovic/Desktop/MunjeLumenDS2025/data/test_truth.csv"
IMAGE_FOLDER = "/Users/mipopovic/Desktop/MunjeLumenDS2025/data/test_output/train/benign/"
MASK_FOLDER = "/Users/mipopovic/Desktop/MunjeLumenDS2025/data/segmentation_mask/"
CSV_OUTPUT = "/Users/mipopovic/Desktop/MunjeLumenDS2025/data/skin_tone_estimates.csv"


process_image_batch(CSV_PATH, IMAGE_FOLDER, MASK_FOLDER, CSV_OUTPUT)
