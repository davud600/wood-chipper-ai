import pandas as pd

from utils import TRAINING_DATA_CSV, TYPES


csv_file = TRAINING_DATA_CSV
data = pd.read_csv(csv_file)

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
}
sum = 0
for type in data["type"].tolist():
    sum += 1
    type_counters[int(type)] += 1

for t, type in enumerate(list(TYPES.keys())):
    print(f"{type}: {type_counters[t]}")
print(f"total: {sum}")
