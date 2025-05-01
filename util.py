import json


def json_dumps(dct: dict):
    return json.dumps(dct, indent=4, sort_keys=True)
