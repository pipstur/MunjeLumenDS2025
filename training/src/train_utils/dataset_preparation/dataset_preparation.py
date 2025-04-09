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


def preprocess_image(args: Tuple[Image.Image, Tuple[int, int], bool, bool, bool]):
    """
    Applies resizing and optional preprocessing steps to an image.
    """
    image, image_size, padding_flag, apply_clahe_flag, remove_hair_flag = args

    image = resize_image(image, image_size, padding_flag)
    if apply_clahe_flag:
        image = apply_clahe(image, 1.4, (8, 8))
    if remove_hair_flag:
        image = remove_hair(image)

    return image


def process_image(args: Tuple[str, str, Tuple[int, int], bool, bool, bool]) -> None:
    """Loads an image, applies preprocessing, and saves the result."""
    src_path, dest_path, image_size, padding_flag, apply_clahe_flag, remove_hair_flag = args
    try:
        with Image.open(src_path) as img:
            img = preprocess_image(
                [img, image_size, padding_flag, apply_clahe_flag, remove_hair_flag]
            )
            img.save(dest_path)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


def process_image_batch(
    csv_file: str,
    image_dir: str,
    output_dir: str,
    image_size: Tuple[int, int],
    padding_flag: bool,
    apply_clahe_flag: bool,
    remove_hair_flag: bool,
) -> None:
    """
    Processes and saves images into 'benign' and 'malignant'
    folders based on labels from CSV file.

    Applies optional preprocessing, resizes images, and saves them
    into categorized folders using multiprocessing.

    Args:
        csv_file (str): Path to CSV with image names and labels.
        image_dir (str): Directory containing input images.
        output_dir (str): Destination directory for processed images.
        image_size (Tuple[int, int]): Target size (width, height) for output images.
        padding_flag (bool): Add padding to preserve aspect ratio if True.
        apply_clahe_flag (bool): Apply CLAHE if True.
        remove_hair_flag (bool): Apply hair removal preprocessing if True.
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
        tasks.append(
            (src_path, dest_path, image_size, padding_flag, apply_clahe_flag, remove_hair_flag)
        )

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
    parser.add_argument(
        "--padding", action="store_true", help="Whether to add padding to scaled image."
    )

    return parser.parse_args()


def main():
    args = cli()
    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError("Output directory already exists. Use --overwrite to replace.")

    process_image_batch(
        args.csv_path,
        args.images_dir,
        args.output_dir,
        tuple(args.image_size),
        args.padding,
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
