import argparse
import os
import shutil
import zipfile

import requests
from tqdm import tqdm


def parse_arguments():
    """Parse command-line arguments for repository details and release tag."""
    parser = argparse.ArgumentParser(description="Download and extract a release from GitHub.")
    parser.add_argument("--repo-owner", type=str, help="GitHub username or organization name.")
    parser.add_argument("--repo-name", type=str, help="GitHub repository name.")
    parser.add_argument("--release-tag", type=str, help="Release tag name to fetch.")
    parser.add_argument(
        "--download-dir",
        type=str,
        default="downloads",
        help="Directory to store downloaded files.",
    )
    parser.add_argument(
        "--extract-dir", type=str, default="extracted", help="Directory to extract files to."
    )
    return parser.parse_args()


def get_release_info(repo_owner, repo_name, release_tag):
    """Fetch release details from the GitHub API for a given repository and release tag,
    then download the file with progress."""
    release_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{release_tag}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.36"
        ),
        "Accept-Encoding": "gzip, deflate, br",
    }

    # Send a request to fetch release details
    response = requests.get(release_url, headers=headers)

    if response.status_code == 200:
        return response.json()

    else:
        print(f"Failed to fetch release information. HTTP status code: {response.status_code}")


def download_zip_file(release_data, download_dir):
    """Download the .zip asset from the provided URL and store it in the specified directory."""
    os.makedirs(download_dir, exist_ok=True)

    for asset in release_data["assets"]:

        if asset["name"].endswith(".zip"):
            download_url = asset["browser_download_url"]
            zip_filename = os.path.join(download_dir, asset["name"])

            file_response = requests.get(download_url, stream=True)
            total_size_in_bytes = int(file_response.headers.get("content-length", 0))

            with tqdm(
                total=total_size_in_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {asset['name']}",
            ) as pbar:
                with open(zip_filename, "wb") as zip_file:
                    for chunk in file_response.iter_content(chunk_size=1024):
                        if chunk:
                            zip_file.write(chunk)
                            pbar.update(len(chunk))

        print(f"Download complete: {zip_filename}")
        return zip_filename
    else:
        print(f"Failed to download {zip_filename}.")
        return None


def extract_zip_file(zip_filename, extract_dir):
    """Extract the downloaded .zip file to the specified extraction directory using zipfile."""
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting {zip_filename} to {extract_dir}...")

    try:
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extraction complete: {zip_filename} to {extract_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_filename} is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred during extraction: {str(e)}")


def clean_up(zip_filename, download_dir):
    """Delete the .zip file and the download directory."""
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        print(f"Deleted {zip_filename}")

    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
        print(f"Deleted '{download_dir}' folder.")


def main():
    """Main function to handle the process of downloading, extracting, and cleaning up."""
    args = parse_arguments()

    release_data = get_release_info(args.repo_owner, args.repo_name, args.release_tag)

    if release_data:
        for asset in release_data["assets"]:
            if asset["name"].endswith(".zip"):

                zip_filename = download_zip_file(release_data, args.download_dir)

                if zip_filename:
                    extract_zip_file(zip_filename, args.extract_dir)

                    clean_up(zip_filename, args.download_dir)


if __name__ == "__main__":
    main()
