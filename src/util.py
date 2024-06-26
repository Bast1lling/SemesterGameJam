import json


def load_json(path: str):
    with open(path, "r") as file:
        return json.load(file)
