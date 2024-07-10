import gdown
import zipfile

def download_and_extract_files(files_to_download):
    def download_file(file_id, output_file):
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, output_file, quiet=False)

    for file in files_to_download:
        download_file(file['id'], file['name'])

    with zipfile.ZipFile('Dataset.zip', 'r') as zip_ref:
        zip_ref.extractall("Data/")