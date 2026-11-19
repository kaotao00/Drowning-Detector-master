import os
import tempfile

import requests


def download_file(url, file_name, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, file_name)
    if os.path.exists(target):
        return target

    print('Downloading ' + file_name + ' from ' + url, flush=True)
    temp_path = None
    try:
        with requests.get(url, allow_redirects=True, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(mode='wb', dir=dest_dir, delete=False) as output:
                temp_path = output.name
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        output.write(chunk)
        os.replace(temp_path, target)
        print('Downloaded ' + file_name, flush=True)
        return target
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
