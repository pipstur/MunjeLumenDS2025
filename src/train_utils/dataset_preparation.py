import os
import argparse
import pandas as pd
from PIL import Image
import shutil
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple, List


def process_image(args: Tuple[str, str, Tuple[int, int]]) -> None:
    """Helper function to resize an image and save it in the appropriate directory."""
    src_path, dest_path, image_size = args
    try:
        with Image.open(src_path) as img:
            img = img.resize(image_size, Image.BILINEAR)
            img.save(dest_path)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def resize_and_save_images(
    csv_file: str, image_dir: str, output_dir: str, image_size: Tuple[int, int]
) -> None:
    """
    Resize images and save them in categorized folders based on labels from a CSV file,
    using multiprocessing for faster processing.

    Args:
        csv_file (str): Path to the CSV file containing image names and labels.
        image_dir (str): Path to the directory containing original images.
        output_dir (str): Path to the directory where resized images will be saved.
        image_size (Tuple[int, int]): Desired image size (width, height) for resizing.
    """
    df = pd.read_csv(csv_file)
    df["image_name"] = df["image_name"] + ".jpg"

    benign_dir = os.path.join(output_dir, "benign")
    malignant_dir = os.path.join(output_dir, "malignant")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    tasks: List[Tuple[str, str, Tuple[int, int]]] = []

    for _, row in df.iterrows():
        dest_dir = benign_dir if row["benign_malignant"] == "benign" else malignant_dir
        src_path = os.path.join(image_dir, row["image_name"])
        dest_path = os.path.join(dest_dir, row["image_name"])
        tasks.append((src_path, dest_path, image_size))

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Resizing images"))


def split_dataset(
    dataset_dir: str, train_dir: str, val_dir: str, val_split: float = 0.2, seed: int = 27
) -> None:
    """
    Split dataset into training and validation sets.

    Args:
        dataset_dir (str): Path to the directory containing categorized images.
        train_dir (str): Path to the directory where training images will be stored.
        val_dir (str): Path to the directory where validation images will be stored.
        val_split (float, optional): Proportion of images to allocate to the validation set
                                    (default is 0.2).
        seed (int, optional): Random seed for reproducibility (default is 27).
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    categories = ["benign", "malignant"]
    random.seed(seed)

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)

        if os.path.isdir(category_path):
            os.makedirs(train_category_path, exist_ok=True)
            os.makedirs(val_category_path, exist_ok=True)

            images = os.listdir(category_path)
            random.shuffle(images)

            val_size = int(len(images) * val_split)
            val_images = images[:val_size]
            train_images = images[val_size:]

            for img in tqdm(train_images, desc=f"Moving {category} train images"):
                shutil.move(
                    os.path.join(category_path, img), os.path.join(train_category_path, img)
                )
            for img in tqdm(val_images, desc=f"Moving {category} val images"):
                shutil.move(os.path.join(category_path, img), os.path.join(val_category_path, img))

            print(f"{category}: {len(train_images)} train, {len(val_images)} val")

    print("Dataset split complete.")


def cli():
    parser = argparse.ArgumentParser(description="Resize images and split dataset.")
    parser.add_argument(
        "--csv-path", type=str, required=True, help="Path to the CSV file with labels."
    )
    parser.add_argument(
        "--images-dir", type=str, required=True, help="Directory containing input images."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save resized images."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (width height). Default: 224x224",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=27, help="Random seed for dataset split (default: 27)"
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing directory."
    )
    return parser.parse_args()


def main():
    args = cli()
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError("Output directory already exists. Use --overwrite to replace.")

    resize_and_save_images(args.csv_path, args.images_dir, args.output_dir, tuple(args.image_size))
    split_dataset(
        args.output_dir,
        os.path.join(args.output_dir, "train"),
        os.path.join(args.output_dir, "val"),
        args.val_split,
        args.seed,
    )


if __name__ == "__main__":
    main()
