import pandas as pd

from src.utils import TRAINING_DATA_CSV, DOCUMENT_TYPES


total_pages = 0
type_counters = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
}
sum = 0


csv_files = [
    # f"{TRAINING_DATA_CSV.replace('.csv', '')}-1.csv",
    # f"{TRAINING_DATA_CSV.replace('.csv', '')}-2.csv",
    TRAINING_DATA_CSV
]

for csv_file in csv_files:
    data = pd.read_csv(csv_file)

    for type in data["type"].tolist():
        sum += 1
        type_counters[int(type)] += 1

for t, type in enumerate(list(DOCUMENT_TYPES.keys())):
    print(f"{type}: {type_counters[t]}")

print(f"total: {sum}")
