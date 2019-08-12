import time
from pathlib import Path


def get_new_checkpoint_folder(folder_path: Path):
    return folder_path / time.strftime('%Y%m%d-%H%M%S')


def get_latest_checkpoint(folder_path: Path):
    latest_checkpoint: Path = None
    latest_timestamp: float = None
    subfolders = [subdir for subdir in folder_path.iterdir() if subdir.is_dir()]

    for subfolder in subfolders:
        struct_time = time.strptime(subfolder.name, '%Y%m%d-%H%M%S')
        timestamp = time.mktime(struct_time)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_checkpoint = subfolder
            latest_timestamp = timestamp

    return latest_checkpoint


if __name__ == '__main__':
    round_folder_path = Path(__file__).parent / r"trained_models/7657-2F43-9B3A-C1F1/1"
    print(get_latest_checkpoint(round_folder_path, "critic*"))
