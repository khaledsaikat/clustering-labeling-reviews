import json


def load_json_from_file(file_path):
    """Load json from a json file"""
    with open(file_path) as json_data:
        return json.load(json_data)


def write_json_to_file(data, file_path):
    """Write json to file"""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)


def load_json_by_line(file_path: str, max_items: int = 100):
    """Load reviews from a json file and return as a generator."""
    with open(file_path) as json_file:
        item_count = 0
        for line in json_file:
            item_count += 1
            if item_count > max_items:
                return
            yield json.loads(line)
