import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "sales_prediction_dataset.json"


def load_data(path=DATA_PATH):
    data = pd.read_json(path)
    return pd.DataFrame(data)
