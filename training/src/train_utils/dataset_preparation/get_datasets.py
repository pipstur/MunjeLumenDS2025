import argparse
import os
import shutil
import zipfile
from typing import List

import pandas as pd
import requests
from tqdm import tqdm


def download_file(url: str, output_path: str) -> None:
    """
    Downloads a file from a given URL to the specified output path with a progress bar.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local file path where the downloaded file will be saved.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as file, tqdm(
            desc=f"Downloading {os.path.basename(output_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        exit(1)


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extracts a ZIP file to a specified directory and moves all images directly to `extract_to`.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_to (str): Destination directory for extracted files.
    """
    temp_extract_dir = os.path.join(extract_to, "temp_extract")
    os.makedirs(temp_extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        exit(1)

    for root, _, files in os.walk(temp_extract_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(extract_to, file)
            shutil.move(src_path, dest_path)

    shutil.rmtree(temp_extract_dir)
    print(f"Extracted and cleaned: {zip_path}")


def process_csv_files(input_dir: str, output_file: str) -> None:
    """
    Processes multiple CSV files, merging them into a single standardized CSV file.

    Args:
        input_dir (str): Directory containing CSV files.
        output_file (str): Output path for the merged CSV file.
    """
    all_data = []
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        # Identify the image column
        image_col = next((col for col in df.columns if "image" in col.lower()), None)
        if not image_col:
            print(f"Warning: No image column found in {file}. Skipping...")
            continue

        # Identify the label column (MEL or benign_malignant)
        label_col = (
            "benign_malignant"
            if "benign_malignant" in df.columns
            else "MEL" if "MEL" in df.columns else None
        )
        if not label_col:
            print(f"Warning: No valid label column found in {file}. Skipping...")
            continue

        # Standardize column names
        df = df[[image_col, label_col]]
        df.columns = ["image_name", "benign_malignant"]

        # Convert MEL column to benign/malignant labels if necessary
        if label_col == "MEL":
            df["benign_malignant"] = df["benign_malignant"].map({0: "benign", 1: "malignant"})

        all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved as {output_file}")
    else:
        print("No valid CSV files were processed.")


def main(dataset_urls: List[str], label_urls: List[str], output_dir: str) -> None:
    """
    Orchestrates downloading, extracting, and processing of dataset and label files.

    Args:
        dataset_urls (List[str]): List of dataset ZIP file URLs.
        label_urls (List[str]): List of label CSV file URLs.
        output_dir (str): Directory where extracted images and labels will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    for i, url in enumerate(dataset_urls):
        zip_path = os.path.join(output_dir, f"dataset_{i}.zip")
        download_file(url, zip_path)
        extract_zip(zip_path, image_dir)
        os.remove(zip_path)

    for i, url in enumerate(label_urls):
        csv_path = os.path.join(label_dir, f"labels_{i}.csv")
        download_file(url, csv_path)

    output_csv = os.path.join(output_dir, "merged_labels.csv")
    process_csv_files(label_dir, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process ISIC dataset.")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for extracted data"
    )
    args = parser.parse_args()
    dataset_urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip",
    ]
    label_urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_GroundTruth.csv",
        "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv",
    ]
    main(dataset_urls, label_urls, args.output_dir)
