from typing import Any, Dict
import argparse
import json
import numpy as np


def write_to_json(
    input_: Dict[str, Any], 
    output_file: str,
    folder: str = "resources"
):

    """
    Write dictionary to .JSON file

    :param input_: dictionaty to be written
    :param output_file: .JSON file name
    """
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    with open(f"{folder}/{output_file}.json", "w") as f:
        json.dump(input_, f, indent=4, cls=CustomEncoder)

    print(f"{output_file} written under {folder}.")

