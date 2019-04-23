import time
from pathlib import Path


def get_model_savepath(folder_path: Path, filename: str, use_timestamp: bool = True):
    if use_timestamp:
        filename += f"_{time.strftime('%Y%m%d-%H%M%S')}"
    return folder_path / (filename + ".h5")


def get_latest_model(folder_path: Path, glob: str = "**/*.h5"):
    latest_model: Path = None
    latest_timestamp: float = None

    for file in folder_path.glob(glob):
        struct_time = time.strptime(file.name[-18:-3], '%Y%m%d-%H%M%S')
        timestamp = time.mktime(struct_time)
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_model = file
            latest_timestamp = timestamp

    return latest_model


if __name__ == '__main__':
    round_folder_path = Path(__file__).parent / r"trained_models/7657-2F43-9B3A-C1F1/1"
    print(get_latest_model(round_folder_path, "critic*"))
