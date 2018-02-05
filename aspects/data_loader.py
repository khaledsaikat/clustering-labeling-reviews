#!/usr/bin/env python3
import json
import re


def loadJsonFromFile(file_path):
    """Load json from a json file"""
    with open(file_path) as json_data:
        return json.load(json_data)


def loadJsonByLine(file_path):
    """Load json from file and return as list.
    Each line contains single json string.
    """

    with open(file_path) as json_file:
        return [json.loads(line) for line in json_file]


def load_json_by_line(file_path: str, max_items: int = 100):
    """Load reviews from a json file and return as a generator."""
    with open(file_path) as json_file:
        item_count = 0
        for line in json_file:
            item_count += 1
            if item_count > max_items:
                return
            yield json.loads(line)


def loadJsonByBraces(file_path):
    """Load json from file and return as list.
    Json is seperated by curly brackets. Consider only simple json
    """

    with open(file_path) as json_file:
        return [json.loads("{"+json_content) for json_content in json_file.read().split("{") if len(json_content)>0]


def loadJsonByAspectBraces(file_path):
    """Load json from file and return as list.
    Json is seperated by curly brackets contaning "aspects" as dictionary
    """

    with open(file_path) as json_file:
        return [json.loads(json_content) for json_content in re.findall(r"\{.*?aspects.*?\}.*?\}", json_file.read(), re.DOTALL)]


def writeJsonByLine(json_list, file_path):
    """Write to a file, each line has single json"""
    with open(file_path, "w") as json_file:
        for single_json in json_list:
            json_file.write(json.dumps(single_json) + "\n")


def writeJsonToFile(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
