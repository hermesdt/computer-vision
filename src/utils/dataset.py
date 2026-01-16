import os
import requests
from pathlib import Path

DATASETS = {
    "imagenette": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

def _download_file(url: str, folder: str) -> str:
    """Download a file from a URL to a local filepath."""

    local_filename = url.split("/")[-1]
    filepath = os.path.join(folder, local_filename)

    # If file already exists, do nothing
    if Path(filepath).exists():
        print(f"File {filepath} already exists. Skipping download.")
        return filepath

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {filepath}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded file from {url} to {filepath}")
    return filepath

def _untar(filepath: str, extract_path: str) -> None:
    """Untar a .tgz file to a specified directory."""
    import tarfile

    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def _unzip(filepath: str, extract_path: str) -> None:
    """Unzip a .zip file to a specified directory."""
    import zipfile

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def download_imagette(folder: str) -> str:
    """Download the Imagenette dataset to a specified folder."""
    url = DATASETS["imagenette"]
    local_filepath = _download_file(url, folder)

    imagenette_folder = os.path.join(folder, "imagenette")
    _untar(local_filepath, imagenette_folder)
    print(f"Decompressed dataset to {imagenette_folder}")

    return imagenette_folder

def download(dataset: str, folder: str) -> str:
    """Download a dataset by name."""
    if dataset == "imagenette":
        return download_imagette(folder)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")
