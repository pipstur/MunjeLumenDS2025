import argparse
import os
import random
import shutil

from tqdm import tqdm


def sample_balanced_dataset(
    benign_dir: str, malignant_dir: str, output_dir: str, seed: int = 27
) -> None:
    """
    Samples all images from the malignant class and an equal number of random images from the
    benign class.

    Args:
        benign_dir (str): Path to the directory containing benign images.
        malignant_dir (str): Path to the directory containing malignant images.
        output_dir (str): Path to the directory where the sampled dataset will be saved.
        seed (int, optional): Random seed for reproducibility (default is 27).
    """
    random.seed(seed)

    malignant_images = os.listdir(malignant_dir)
    num_malignant = len(malignant_images)

    benign_images = os.listdir(benign_dir)
    benign_sample = random.sample(benign_images, min(num_malignant, len(benign_images)))

    benign_output = os.path.join(output_dir, "benign")
    malignant_output = os.path.join(output_dir, "malignant")
    os.makedirs(benign_output, exist_ok=True)
    os.makedirs(malignant_output, exist_ok=True)

    print(f"Copying {num_malignant} malignant images...")
    for img in tqdm(malignant_images):
        shutil.copy(os.path.join(malignant_dir, img), os.path.join(malignant_output, img))

    print(f"Copying {len(benign_sample)} benign images...")
    for img in tqdm(benign_sample):
        shutil.copy(os.path.join(benign_dir, img), os.path.join(benign_output, img))

    print("Sampling complete!")


def cli():
    parser = argparse.ArgumentParser(
        description="Sample equal number of benign and malignant images."
    )
    parser.add_argument(
        "--benign-dir", type=str, required=True, help="Path to the benign images directory."
    )
    parser.add_argument(
        "--malignant-dir", type=str, required=True, help="Path to the malignant images directory."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to save the sampled dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=27, help="Random seed for reproducibility (default: 27)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    sample_balanced_dataset(args.benign_dir, args.malignant_dir, args.output_dir, args.seed)
