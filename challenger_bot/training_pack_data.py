import json
from pathlib import Path
from typing import List

__all__ = ['rounds_ball_data']


TRAINING_PACK_JSON_FILENAME = Path("7657-2F43-9B3A-C1F1.json")

json_filepath = Path(__file__).parent / TRAINING_PACK_JSON_FILENAME

# print(json_filepath)

assert json_filepath.is_file(), f"JSON file {json_filepath} is not a file."


with open(json_filepath, 'r') as f:
    training_pack_json = json.load(f)


rounds: List[dict] = training_pack_json['TrainingData']['Rounds']


def get_ball_data(round: dict):
    for obj in round['SerializedArchetypes']:
        obj = json.loads(obj)
        if 'Ball' in obj['ObjectArchetype']:
            return [obj['StartLocationX'], obj['StartLocationY'], obj['StartLocationZ']]


rounds_ball_data = [
    get_ball_data(round)
    for round in rounds
]

