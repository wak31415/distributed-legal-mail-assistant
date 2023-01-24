import pandas as pd
from typing import List
import os

def get_data(path_to_dir: str) -> List:
    ans = []
    for dirname, _, filenames in os.walk(path_to_dir):
        for fname in filenames:
            with open(os.path.join(dirname,fname), 'r') as f:
                lines = f.readlines()
            labels, email = lines[0], ''.join(lines[1:])
            labels = [x.strip() for x in labels.split(",")]
            ans.append({"text": email, "labels": labels})
    return ans