import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".pre-commit-config.yaml", ".git", ".github"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from img_utils.preprocessing import apply_clahe, resize_image

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_PADDING_COLOR = (255, 255, 255)
CLAHE_CLIP_LIMIT = 1.4
CLAHE_TILE_GRID_SIZE = (8, 8)


def load_onnx_models(models_folder: str) -> List[ort.InferenceSession]:
    """Load ONNX models from a given folder."""
    sessions = []
    for root, _, files in os.walk(models_folder):
        for file in files:
            if file.endswith(".onnx"):
                model_path = os.path.join(root, file)
                session = ort.InferenceSession(model_path, providers=PROVIDERS)
                sessions.append(session)
    return sessions


def preprocess_image(
    image_path: str, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, padding: bool = True
) -> Image.Image:
    """Preprocess the input image: resize and apply CLAHE."""
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image, image_size, padding)
    image = apply_clahe(image, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE)
    return image


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


def soft_vote(predictions: List[np.ndarray]) -> Tuple[int, float]:
    """Aggregate predictions using soft voting.

    Returns:
        - final_prediction: int (the predicted class)
        - confidence: float (the probability of the selected class)
    """
    probabilities = [np.exp(logits) / np.sum(np.exp(logits)) for logits in predictions]
    avg_prob = np.mean(probabilities, axis=0)

    final_prediction = int(np.argmax(avg_prob))
    confidence = float(avg_prob[final_prediction])
    if final_prediction == 0:
        confidence = 1 - confidence
    return final_prediction, confidence


def main(input_folder, models_folder, output_csv, save_tiles_folder=None):
    print("Loading models...")
    sessions = load_onnx_models(models_folder)
    if not sessions:
        print("No ONNX models found in the models folder.")
        return

    print(f"Loaded {len(sessions)} models.")
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
        final_prediction, confidence = soft_vote(predictions)

        results.append(
            {
                "image_name": original_filename.replace(".jpg", ""),
                "target": final_prediction,
                "confidence": confidence,
            }
        )

    with open(output_csv, mode="w", newline="") as csv_file:
        fieldnames = ["image_name", "target", "confidence"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Model Inference with Majority Voting")
    parser.add_argument("input_folder", type=str, help="Folder containing input images")
    parser.add_argument("output_csv", type=str, help="Path to save the output CSV")
    parser.add_argument(
        "--models-folder", type=str, default="models/", help="Folder containing ONNX models"
    )
    parser.add_argument(
        "--save-tiles-folder",
        type=str,
        help="Folder to save preprocessed image tiles",
    )
    args = parser.parse_args()

    main(
        args.input_folder,
        args.models_folder,
        args.output_csv,
        args.save_tiles_folder,
    )
