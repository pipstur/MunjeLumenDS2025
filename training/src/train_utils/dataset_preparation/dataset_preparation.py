import argparse
import os
import random
import shutil
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm


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


def remove_hair(image: Image.Image) -> Image.Image:
    """
    Removes hair artifacts from a PIL image using morphological operations and inpainting.
    It performs the following steps:
    - Converts the image to grayscale.
    - Applies a black hat morphological operation to highlight hair strands.
    - Blurs the result to smooth noise.
    - Creates a binary mask by thresholding.
    - Uses inpainting to remove the hair from the original image based on the mask.

    Args:
        image (PIL.Image.Image): Input image as a PIL Image in RGB format.

    Returns:
        PIL.Image.Image: A new PIL Image with hair artifacts removed.
    """
    image_np = np.array(image)
    grayScale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (3, 3), 0)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(image_np, mask, 3, cv2.INPAINT_TELEA)
    cleaned_pil = Image.fromarray(cleaned)

    return cleaned_pil


def process_image(args: Tuple[str, str, Tuple[int, int], bool, bool]) -> None:
    """Helper function to resize an image and save it in the appropriate directory."""
    src_path, dest_path, image_size, apply_clahe_flag, remove_hair_flag = args
    try:
        with Image.open(src_path) as img:
            img = img.resize(image_size, Image.BILINEAR)
            if apply_clahe_flag:
                img = apply_clahe(img)
            if remove_hair_flag:
                img = remove_hair(img)
            img.save(dest_path)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def resize_and_save_images(
    csv_file: str,
    image_dir: str,
    output_dir: str,
    image_size: Tuple[int, int],
    apply_clahe_flag: bool,
    remove_hair_flag: bool,
) -> None:
    """
    Resize images and save them in categorized folders based on labels from a CSV file.
    """
    df = pd.read_csv(csv_file)
    df["image_name"] = df["image_name"] + ".jpg"

    benign_dir = os.path.join(output_dir, "benign")
    malignant_dir = os.path.join(output_dir, "malignant")
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    tasks: List[Tuple[str, str, Tuple[int, int]], bool] = []
    for _, row in df.iterrows():
        dest_dir = benign_dir if row["benign_malignant"] == "benign" else malignant_dir
        src_path = os.path.join(image_dir, row["image_name"])
        dest_path = os.path.join(dest_dir, row["image_name"])
        tasks.append((src_path, dest_path, image_size, apply_clahe_flag, remove_hair_flag))

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Processing images"))


def standard_split(
    dataset_dir: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    """
    Perform a standard train/val/test split.

    Args:
        dataset_dir (str): Path to the dataset containing 'benign' and 'malignant' folders.
        train_dir (str): Path to save the training dataset.
        val_dir (str): Path to save the validation dataset.
        test_dir (str): Path to save the test dataset.
        val_split (float): Fraction of images to be used for validation.
        test_split (float): Fraction of images to be used for testing.
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    categories = ["benign", "malignant"]
    random.seed(seed)

    for cat in categories:
        category_path = os.path.join(dataset_dir, cat)
        train_category_path = os.path.join(train_dir, cat)
        val_category_path = os.path.join(val_dir, cat)
        test_category_path = os.path.join(test_dir, cat)

        if os.path.isdir(category_path):
            os.makedirs(train_category_path, exist_ok=True)
            os.makedirs(val_category_path, exist_ok=True)
            os.makedirs(test_category_path, exist_ok=True)

            images = os.listdir(category_path)
            random.shuffle(images)

            total = len(images)
            test_size = int(total * test_split)
            val_size = int(total * val_split)

            test_images = images[:test_size]
            val_images = images[test_size : test_size + val_size]
            train_images = images[test_size + val_size :]

            for img in train_images:
                shutil.move(
                    os.path.join(category_path, img), os.path.join(train_category_path, img)
                )
            for img in val_images:
                shutil.move(os.path.join(category_path, img), os.path.join(val_category_path, img))
            for img in test_images:
                shutil.move(
                    os.path.join(category_path, img), os.path.join(test_category_path, img)
                )

            print(
                f"{cat}:{len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
            )

            shutil.rmtree(category_path)

    print("Dataset split complete.")


def kfold_split(dataset_dir: str, output_dir: str, n_splits: int, seed: int) -> None:
    """
    Perform stratified k-fold split with separate test sets for each fold.

    Args:
        dataset_dir (str): Path to dataset containing "benign" and "malignant" folders.
        output_dir (str): Path where the k-fold splits will be stored.
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
    """
    df = []
    for category in ["benign", "malignant"]:
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            labels = [category] * len(images)
            df.extend(zip(images, labels))

    df = pd.DataFrame(df, columns=["image_name", "label"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df["image_name"], df["label"])):
        fold_dir = os.path.join(output_dir, f"fold{fold+1}")
        train_dir, val_dir, test_dir = (
            os.path.join(fold_dir, "train"),
            os.path.join(fold_dir, "val"),
            os.path.join(fold_dir, "test"),
        )

        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(d, "benign"), exist_ok=True)
            os.makedirs(os.path.join(d, "malignant"), exist_ok=True)

        # Split train_val further into training and validation sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(
            sss.split(df.iloc[train_val_idx]["image_name"], df.iloc[train_val_idx]["label"])
        )

        # Move test images
        for img_name, label in tqdm(
            df.iloc[test_idx][["image_name", "label"]].values,
            desc=f"Processing fold {fold+1} (test)",
            leave=False,
        ):
            src_path = os.path.join(dataset_dir, label, img_name)
            dest_path = os.path.join(test_dir, label, img_name)
            shutil.copy(src_path, dest_path)

        for indices, split_dir, split_name in [
            (train_idx, train_dir, "train"),
            (val_idx, val_dir, "val"),
        ]:
            for img_name, label in tqdm(
                df.iloc[train_val_idx].iloc[indices][["image_name", "label"]].values,
                desc=f"Processing fold {fold+1} ({split_name})",
                leave=False,
            ):
                src_path = os.path.join(dataset_dir, label, img_name)
                dest_path = os.path.join(split_dir, label, img_name)
                shutil.copy(src_path, dest_path)

        print(f"Fold {fold+1}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    shutil.rmtree(os.path.join(output_dir, "benign"))
    shutil.rmtree(os.path.join(output_dir, "malignant"))
    print("K-Fold dataset split complete.")


def cli():
    parser = argparse.ArgumentParser(description="Dataset preparation for the training of models.")
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
        "--split-type",
        type=str,
        required=True,
        choices=["train", "kfold", "train-val-test"],
        help="Type of split to perform.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Fraction of data for validation."
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Fraction of data for testing."
    )
    parser.add_argument("--seed", type=int, default=27, help="Random seed for dataset split.")
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite existing directory."
    )
    parser.add_argument(
        "--kfold", type=int, default=None, help="Number of folds for K-Fold cross-validation."
    )
    parser.add_argument(
        "--apply-clahe", action="store_true", help="Whether to apply CLAHE enhancment."
    )
    parser.add_argument(
        "--remove-hair", action="store_true", help="Whether to apply hair removal."
    )

    return parser.parse_args()


def main():
    args = cli()
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError("Output directory already exists. Use --overwrite to replace.")

    resize_and_save_images(
        args.csv_path,
        args.images_dir,
        args.output_dir,
        tuple(args.image_size),
        args.apply_clahe,
        args.remove_hair,
    )

    if args.split_type == "train":
        standard_split(
            args.output_dir,
            os.path.join(args.output_dir, "train"),
            os.path.join(args.output_dir, "train"),
            os.path.join(args.output_dir, "train"),
            0,
            0,
            args.seed,
        )
    elif args.split_type == "train-val-test":
        standard_split(
            args.output_dir,
            os.path.join(args.output_dir, "train"),
            os.path.join(args.output_dir, "val"),
            os.path.join(args.output_dir, "test"),
            args.val_split,
            args.test_split,
            args.seed,
        )
    elif args.kfold:
        kfold_split(args.output_dir, args.output_dir, args.kfold, args.seed)


if __name__ == "__main__":
    main()
