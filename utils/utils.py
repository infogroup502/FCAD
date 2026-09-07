import os


def mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
