import numpy as np
import fastText as ft
from pathlib import Path
import requests
from tqdm import tqdm



def assemble_emb_mat(vocab, ft_emb_path):
    model = ft.load_model(ft_emb_path)
    emb_dim = model.get_dimension()
    emb_mat = np.zeros([len(vocab), emb_dim], dtype=np.float32)
    tokens = [token for token in vocab]
    for n, token in enumerate(tokens):
        emb_mat[n] = model.get_word_vector(token)
    return emb_mat


def download(dest_file_path, source_url, force_download=True):
    """Download a file from URL to one or several target locations

    Args:
        dest_file_path: path or list of paths to the file destination files (including file name)
        source_url: the source URL
        force_download: download file if it already exists, or not

    """
    CHUNK = 16 * 1024

    if isinstance(dest_file_path, str):
        dest_file_path = [Path(dest_file_path).absolute()]
    elif isinstance(dest_file_path, Path):
        dest_file_path = [dest_file_path.absolute()]
    elif isinstance(dest_file_path, list):
        dest_file_path = [Path(path) for path in dest_file_path]

    first_dest_path = dest_file_path.pop()

    if force_download or not first_dest_path.exists():
        first_dest_path.parent.mkdir(parents=True, exist_ok=True)

        r = requests.get(source_url, stream=True)
        total_length = int(r.headers.get('content-length', 0))

        with first_dest_path.open('wb') as f:
            print('Downloading from {} to {}'.format(source_url, first_dest_path))

            pbar = tqdm(total=total_length, unit='B', unit_scale=True)
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            f.close()
    else:
        print('File already exists in {}'.format(first_dest_path))

    while len(dest_file_path) > 0:
        dest_path = dest_file_path.pop()

        if force_download or not dest_path.exists():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(first_dest_path), str(dest_path))
        else:
            print('File already exists in {}'.format(dest_path))
