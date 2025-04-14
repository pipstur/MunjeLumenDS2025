import argparse
import csv
import os
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_PADDING_COLOR = (255, 255, 255)
CLAHE_CLIP_LIMIT = 1.0
CLAHE_TILE_GRID_SIZE = (5, 5)


def resize_image(
    image: Image.Image,
    image_size: Tuple[int, int],
    padding_flag: bool,
    padding_color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR,
) -> Image.Image:
    """Resize an image with optional padding to preserve aspect ratio."""
    if padding_flag:
        canvas_size = image_size
        image.thumbnail(canvas_size, Image.LANCZOS)
        canvas = Image.new("RGB", canvas_size, padding_color)
        paste_x = (canvas_size[0] - image.size[0]) // 2
        paste_y = (canvas_size[1] - image.size[1]) // 2
        canvas.paste(image, (paste_x, paste_y))
        return canvas
    else:
        return image.resize(image_size, Image.BILINEAR)


def apply_clahe(
    image: Image.Image,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> Image.Image:
    """Apply CLAHE to enhance the contrast of the image."""
    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return Image.fromarray(enhanced_image)


def load_onnx_models(models_folder: str) -> List[ort.InferenceSession]:
    """Load ONNX models from a given folder."""
    sessions = []
    for model_name in os.listdir(models_folder):
        if model_name.endswith(".onnx"):
            model_path = os.path.join(models_folder, model_name)
            session = ort.InferenceSession(model_path, providers=PROVIDERS)
            sessions.append(session)
    return sessions


def preprocess_image(
    image_path: str, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, padding: bool = True
) -> Image.Image:
    """Preprocess the input image: resize and apply CLAHE."""
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image, image_size, padding)
    return apply_clahe(image)


def save_preprocessed_image(
    image: Image.Image, output_folder: str, original_filename: str
) -> None:
    """Save the preprocessed image to a specified folder."""
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, original_filename)
    image.save(save_path)


def prepare_image_tensor(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a normalized tensor suitable for model input."""
    transform_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform_pipeline(image).numpy()
    return np.expand_dims(tensor, axis=0)


def predict_image(sessions: List[ort.InferenceSession], image_tensor: np.ndarray) -> List[float]:
    """Run inference on the image using all loaded models."""
    predictions = []
    for session in sessions:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_tensor})
        prob = outputs[0].squeeze()
        predictions.append(prob)
    return predictions


def soft_vote(predictions: List[np.ndarray]) -> int:
    """Aggregate predictions using soft voting."""
    probabilities = [np.exp(logits) / np.sum(np.exp(logits)) for logits in predictions]

    avg_prob = np.mean(probabilities, axis=0)
    final_prediction = int(np.argmax(avg_prob))

    return final_prediction


def majority_vote(predictions: List[int]) -> int:
    """Aggregate predictions using majority voting."""
    class_predictions = [int(np.argmax(p)) for p in predictions]
    count = Counter(class_predictions)
    return count.most_common(1)[0][0]


def main(input_folder, models_folder, output_csv, soft_voting, save_tiles_folder=None):
    print("Loading models...")
    sessions = load_onnx_models(models_folder)
    if not sessions:
        print("No ONNX models found in the models folder.")
        return

    image_paths = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        original_filename = os.path.basename(image_path)

        image = preprocess_image(image_path)

        if save_tiles_folder:
            save_preprocessed_image(image, save_tiles_folder, original_filename)

        image_tensor = prepare_image_tensor(image)

        predictions = predict_image(sessions, image_tensor)
        if soft_voting:
            final_prediction = soft_vote(predictions)
        else:
            final_prediction = majority_vote(predictions)

        benign_malignant = "benign" if final_prediction == 0 else "malignant"

        results.append(
            {
                "image_name": original_filename.replace(".jpg", ""),
                "benign_malignant": benign_malignant,
            }
        )

    with open(output_csv, mode="w", newline="") as csv_file:
        fieldnames = ["image_name", "benign_malignant"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Inference with Majority Voting")
    parser.add_argument(
        "--input-folder", type=str, required=True, help="Folder containing input images"
    )
    parser.add_argument(
        "--models-folder", type=str, default="models/", help="Folder containing ONNX models"
    )
    parser.add_argument(
        "--output-csv", type=str, default="results.csv", help="Path to save the output CSV"
    )
    parser.add_argument(
        "--save-tiles-folder",
        type=str,
        help="Folder to save preprocessed image tiles",
    )
    parser.add_argument("--soft-vote", action="store_true", help="Use soft voting for predictions")
    args = parser.parse_args()

    main(
        args.input_folder,
        args.models_folder,
        args.output_csv,
        args.soft_vote,
        args.save_tiles_folder,
    )
