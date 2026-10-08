import pandas as pd
from pathlib import Path

def load_data():
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "Dataset" / "Superstore.csv"

    # Use the first available CSV when the default Superstore file is absent.
    if not csv_path.exists():
        csv_path = list((base_path / "Dataset").glob("*.csv"))[0]

    data = pd.read_csv(csv_path)

    # Normalize column names before case-insensitive lookup.
    data.columns = data.columns.str.strip()
    col_map = {c.lower(): c for c in data.columns}

    # Derive Aura from profit margin when both financial fields are available.
    if 'profit' in col_map and 'sales' in col_map:
        data["Aura"] = data[col_map['profit']] / data[col_map['sales']] * 100
    elif 'sales' in col_map:
        data["Aura"] = data[col_map['sales']] * 0.1 # Apply a temporary 10% estimate when profit data is unavailable.
    else:
        data["Aura"] = 0

    return data