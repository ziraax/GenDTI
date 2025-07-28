import os 
import zipfile 
import requests
from tqdm import tqdm

BINDINGDB_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202507_tsv.zip"
DEST_DIR = "data/raw"
DEST_ZIP = os.path.join(DEST_DIR, "BindingDB.zip")
EXTRACTED_FILE = os.path.join(DEST_DIR, "BindingDB.tsv")

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"Extracted files to {extract_dir}")

if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    
    if not os.path.exists(DEST_ZIP):
        download_file(BINDINGDB_URL, DEST_ZIP)
    else:
        print(f"{DEST_ZIP} already exists. Skipping download.")
    
    extract_zip(DEST_ZIP, DEST_DIR)


